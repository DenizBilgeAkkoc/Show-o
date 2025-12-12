# coding=utf-8
# Copyright 2025 NUS Show Lab.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
MathCanvas Dataset for interleaved visual-textual mathematical reasoning.
Based on the MathCanvas-Instruct dataset from the paper:
"MathCanvas: Intrinsic Visual Chain-of-Thought for Multimodal Mathematical Reasoning"
"""

import collections
import random
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple

import pyarrow.parquet as pq
import torch
from datasets.utils import image_transform, format_interleaved_sequence
from PIL import Image
from torch.utils.data import Dataset


class MathCanvasDataset(Dataset):
    """Dataset for MathCanvas interleaved visual-text mathematical reasoning."""

    def __init__(
            self,
            parquet_path: str,
            text_tokenizer: Any,
            max_seq_len: int = 3840,
            image_size: int = 384,
            latent_height: int = 24,
            latent_width: int = 24,
            num_image_tokens: int = 576,
            cond_dropout_prob: float = 0.1,
            max_num_images: int = 4,
            showo_token_ids: Optional[Dict[str, int]] = None,
            system: Tuple[str, str, str] = ("", "", ""),
            min_res: Optional[Tuple[int, int]] = None,
            include_question_images: bool = True,
            include_solution_images: bool = True,
    ) -> None:
        """Initializes the MathCanvas dataset.

        Args:
            parquet_path: Path to the parquet file containing the dataset.
            text_tokenizer: Tokenizer for text processing.
            max_seq_len: Maximum sequence length.
            image_size: Size to which images are resized.
            latent_height: Height of latent representation.
            latent_width: Width of latent representation.
            num_image_tokens: Number of tokens representing an image.
            cond_dropout_prob: Probability of conditioning dropout.
            max_num_images: Maximum number of images per sample.
            showo_token_ids: Dictionary of special token IDs.
            system: Tuple of system prompt strings.
            min_res: Minimum resolution (height, width) for images.
            include_question_images: Whether to include images from questions.
            include_solution_images: Whether to include images from solutions.
        """
        self.text_tokenizer = text_tokenizer
        self.pad_id = self.text_tokenizer.pad_token_id
        self.bos_id = showo_token_ids['bos_id']
        self.eos_id = showo_token_ids['eos_id']
        self.boi_id = showo_token_ids['boi_id']
        self.eoi_id = showo_token_ids['eoi_id']
        self.img_pad_id = showo_token_ids['img_pad_id']
        self.max_seq_len = max_seq_len
        self.image_size = image_size
        self.num_image_tokens = num_image_tokens
        self.h = latent_height
        self.w = latent_width
        self.cond_dropout_prob = cond_dropout_prob
        self.data_type = "interleaved_data"
        self.transform = image_transform
        self.max_num_images = max_num_images
        self.include_question_images = include_question_images
        self.include_solution_images = include_solution_images

        self.parquet_path = parquet_path
        self.samples: List[Dict[str, Any]] = []

        # Load the parquet file metadata first
        self.pf = pq.ParquetFile(parquet_path)
        self.num_row_groups = self.pf.metadata.num_row_groups
        
        # Build an index of (row_group_idx, row_idx_within_group) for random access
        self._index = []
        self._row_group_cache = {}
        self._cached_rg_idx = -1
        
        for rg_idx in range(self.num_row_groups):
            rg_metadata = self.pf.metadata.row_group(rg_idx)
            num_rows = rg_metadata.num_rows
            for row_idx in range(num_rows):
                self._index.append((rg_idx, row_idx))
        
        print(f"MathCanvas dataset loaded. {len(self._index)} samples across {self.num_row_groups} row groups!")

        self.flag_tokens = self.text_tokenizer(
            "Mathematical reasoning with visual chain-of-thought.", add_special_tokens=False
        ).input_ids
        self.system_tokens = self.text_tokenizer(system, add_special_tokens=False).input_ids
        self.system_token_len = sum(len(tokens) for tokens in self.system_tokens)

        if len(self.system_tokens[0]) == 0:
            # 4 for bos, eos, boi, and eoi tokens
            self.max_text_len = (
                                        max_seq_len
                                        - len(self.flag_tokens)
                                        - (num_image_tokens + 2) * max_num_images
                                        - 2
                                ) // max_num_images
        else:
            # 4 for bos, eos, boi, and eoi tokens
            # 1 for eos after text token (a bit tricky)
            self.max_text_len = (
                                        max_seq_len
                                        - (num_image_tokens + 2) * max_num_images
                                        - 2
                                        - self.system_token_len
                                        - 1
                                ) // max_num_images

        self.min_res = min_res if min_res is not None else (256, 256)

    def _get_pil_image(self, image_data: Dict) -> Image.Image:
        """Convert image data from parquet to PIL Image.
        
        Args:
            image_data: Dictionary containing 'bytes' key with raw image data.
            
        Returns:
            PIL Image object.
        """
        if isinstance(image_data, dict) and 'bytes' in image_data:
            return Image.open(BytesIO(image_data['bytes'])).convert('RGB')
        elif isinstance(image_data, bytes):
            return Image.open(BytesIO(image_data)).convert('RGB')
        else:
            raise TypeError(f"Unsupported image data type: {type(image_data)}")

    def _get_interleaved_data(
            self, row: Dict[str, Any]
    ) -> Tuple[List[Optional[torch.Tensor]], List[Optional[List[int]]], List[str]]:
        """Extracts interleaved image-text data from a parquet row.

        The MathCanvas format has:
        - question_interleave: list of {type: 'text'/'image', content: str, index: int}
        - question_images: list of image dicts
        - solution_interleave: list of {type: 'text'/'image', content: str, index: int}
        - solution_images: list of image dicts

        Args:
            row: A row from the parquet dataframe.

        Returns:
            Tuple of image list, tokenized text list, and raw texts.
        """
        question_interleave = row.get('question_interleave', [])
        question_images = row.get('question_images', [])
        solution_interleave = row.get('solution_interleave', [])
        solution_images = row.get('solution_images', [])

        image_list: List[Optional[torch.Tensor]] = []
        text_token_list: List[Optional[List[int]]] = []
        texts: List[str] = []
        
        # Track total images to enforce max_num_images limit
        total_images = 0

        # Process question part
        current_text = ""
        for item in question_interleave:
            if item['type'] == 'text':
                content = item.get('content', '')
                if content:
                    current_text += content + " "
            elif item['type'] == 'image' and self.include_question_images:
                if total_images >= self.max_num_images:
                    continue
                    
                # First, flush any accumulated text
                if current_text.strip():
                    text_tokens = self.text_tokenizer(
                        current_text.strip(),
                        add_special_tokens=False,
                        truncation=True,
                        max_length=self.max_text_len,
                    ).input_ids
                    text_token_list.append(text_tokens)
                    texts.append(current_text.strip())
                    image_list.append(None)  # No image paired with this text
                    current_text = ""
                
                # Add the image
                img_idx = item.get('index', 0)
                if img_idx < len(question_images):
                    try:
                        image = self._get_pil_image(question_images[img_idx])
                        image_tensor = self.transform(image, resolution=self.image_size)
                        image_list.append(image_tensor)
                        text_token_list.append(None)  # No text paired with this image
                        texts.append('')
                        total_images += 1
                    except Exception as e:
                        print(f"Failed to load question image: {e}")
                        continue

        # Add separator between question and solution
        if current_text.strip():
            current_text += "\n\nSolution: "
        else:
            current_text = "Solution: "

        # Process solution part  
        for item in solution_interleave:
            if item['type'] == 'text':
                content = item.get('content', '')
                if content:
                    current_text += content + " "
            elif item['type'] == 'image' and self.include_solution_images:
                if total_images >= self.max_num_images:
                    continue
                    
                # First, flush any accumulated text
                if current_text.strip():
                    text_tokens = self.text_tokenizer(
                        current_text.strip(),
                        add_special_tokens=False,
                        truncation=True,
                        max_length=self.max_text_len,
                    ).input_ids
                    text_token_list.append(text_tokens)
                    texts.append(current_text.strip())
                    current_text = ""
                
                # Add the image
                img_idx = item.get('index', 0)
                if img_idx < len(solution_images):
                    try:
                        image = self._get_pil_image(solution_images[img_idx])
                        image_tensor = self.transform(image, resolution=self.image_size)
                        image_list.append(image_tensor)
                        text_token_list.append(None)  # Text will come after
                        texts.append('')
                        total_images += 1
                    except Exception as e:
                        print(f"Failed to load solution image: {e}")
                        continue

        # Flush any remaining text
        if current_text.strip():
            # Add final answer
            answer = row.get('answer', '')
            if answer:
                current_text += f"\n\nFinal Answer: {answer}"
            
            text_tokens = self.text_tokenizer(
                current_text.strip(),
                add_special_tokens=False,
                truncation=True,
                max_length=self.max_text_len,
            ).input_ids
            text_token_list.append(text_tokens)
            texts.append(current_text.strip())
            image_list.append(None)

        # Now we need to pair images with text properly
        # The format should be: [text, image, text, image, ...] or similar
        # Reorganize to have text-image pairs
        paired_images: List[Optional[torch.Tensor]] = []
        paired_text_tokens: List[Optional[List[int]]] = []
        paired_texts: List[str] = []
        
        i = 0
        while i < len(image_list):
            if image_list[i] is not None:
                # This is an image, look for preceding text
                paired_images.append(image_list[i])
                if i > 0 and text_token_list[i-1] is not None:
                    paired_text_tokens.append(text_token_list[i-1])
                    paired_texts.append(texts[i-1])
                else:
                    paired_text_tokens.append([])
                    paired_texts.append('')
            elif text_token_list[i] is not None:
                # Check if next is an image
                if i + 1 < len(image_list) and image_list[i+1] is not None:
                    # Will be handled in next iteration
                    pass
                else:
                    # Text without following image
                    paired_text_tokens.append(text_token_list[i])
                    paired_texts.append(texts[i])
                    paired_images.append(None)
            i += 1

        # If we have more text than images, or the structure doesn't align well,
        # fall back to a simpler approach: just pair texts with images in order
        if len(paired_images) == 0 or len(paired_text_tokens) == 0:
            # Fallback: create pairs from non-None elements
            all_images = [img for img in image_list if img is not None]
            all_texts = [(t, txt) for t, txt in zip(text_token_list, texts) if t is not None]
            
            paired_images = []
            paired_text_tokens = []
            paired_texts = []
            
            max_pairs = max(len(all_images), len(all_texts))
            for i in range(max_pairs):
                if i < len(all_texts):
                    paired_text_tokens.append(all_texts[i][0])
                    paired_texts.append(all_texts[i][1])
                else:
                    paired_text_tokens.append([])
                    paired_texts.append('')
                
                if i < len(all_images):
                    paired_images.append(all_images[i])
                else:
                    paired_images.append(None)

        # Add flag token to the first text token list
        if len(paired_text_tokens) > 0 and paired_text_tokens[0] is not None:
            paired_text_tokens[0] = self.flag_tokens + paired_text_tokens[0]

        # Ensure we don't exceed max_num_images
        if len(paired_images) > self.max_num_images:
            paired_images = paired_images[:self.max_num_images]
            paired_text_tokens = paired_text_tokens[:self.max_num_images]
            paired_texts = paired_texts[:self.max_num_images]

        # Pad lists if fewer than max_num_images
        while len(paired_images) < self.max_num_images:
            paired_images.append(None)
            paired_text_tokens.append(None)
            paired_texts.append('')

        return paired_images, paired_text_tokens, paired_texts

    def __len__(self) -> int:
        return len(self._index)

    def _get_row(self, idx: int) -> Dict[str, Any]:
        """Get a row from the parquet file using cached row group loading."""
        rg_idx, row_idx = self._index[idx]
        
        # Check if we need to load a new row group
        if self._cached_rg_idx != rg_idx:
            # Read the row group to pandas (one row group at a time to handle nested data)
            table = self.pf.read_row_group(rg_idx)
            self._row_group_cache = table.to_pydict()
            self._cached_rg_idx = rg_idx
        
        # Extract the row from the cached data
        row = {}
        for key in self._row_group_cache:
            row[key] = self._row_group_cache[key][row_idx]
        
        return row

    def __getitem__(self, idx: int) -> Optional[Dict[str, Any]]:
        try:
            row = self._get_row(idx)
            
            # Check if this sample has any content
            question_interleave = row.get('question_interleave', [])
            solution_interleave = row.get('solution_interleave', [])
            
            if len(question_interleave) == 0 and len(solution_interleave) == 0:
                return self.__getitem__((idx + 1) % len(self))

            image_list, text_token_list, texts = self._get_interleaved_data(row)
            
            # Check if we got valid data
            valid_content = any(t is not None and len(t) > 0 for t in text_token_list) or \
                           any(img is not None for img in image_list)
            if not valid_content:
                return self.__getitem__((idx + 1) % len(self))

            (
                text_tokens,
                text_labels,
                modality_positions,
                text_mask,
                image_mask,
            ) = format_interleaved_sequence(
                image_list,
                text_token_list,
                self.bos_id,
                self.eos_id,
                self.boi_id,
                self.eoi_id,
                self.pad_id,
                self.img_pad_id,
                self.num_image_tokens,
                self.max_seq_len,
                self.max_num_images,
            )

            # Ignore flag tokens in the label (first one is bos token)
            text_labels[1: len(self.flag_tokens) + 1] = -100

            # Stack images, using zeros for None images
            temp: List[torch.Tensor] = []
            for img in image_list:
                if img is not None:
                    temp.append(img)
                else:
                    temp.append(torch.zeros((3, self.image_size, self.image_size)))

            image = torch.stack(temp, dim=0)
            
            return {
                'text_tokens': text_tokens,
                'text_labels': text_labels,
                'images': image,
                'modality_positions': modality_positions,
                'text_masks': text_mask,
                'image_masks': image_mask,
                'texts': texts,
                'data_type': self.data_type,
            }

        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            return self.__getitem__((idx + 1) % len(self))

    def collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Collate function to batch data."""
        batched = collections.defaultdict(list)
        for data in batch:
            for key, value in data.items():
                batched[key].append(value)
        for key, value in batched.items():
            if key not in ('texts', 'data_type'):
                batched[key] = torch.stack(value, dim=0)
        return batched


class MathCanvasIterableDataset(torch.utils.data.IterableDataset):
    """Iterable version of MathCanvas dataset for large parquet files.
    
    This version streams data from the parquet file instead of loading all into memory.
    """

    def __init__(
            self,
            parquet_path: str,
            text_tokenizer: Any,
            max_seq_len: int = 3840,
            image_size: int = 384,
            latent_height: int = 24,
            latent_width: int = 24,
            num_image_tokens: int = 576,
            cond_dropout_prob: float = 0.1,
            max_num_images: int = 4,
            showo_token_ids: Optional[Dict[str, int]] = None,
            system: Tuple[str, str, str] = ("", "", ""),
            min_res: Optional[Tuple[int, int]] = None,
            include_question_images: bool = True,
            include_solution_images: bool = True,
            shuffle: bool = True,
    ) -> None:
        """Initialize the iterable MathCanvas dataset."""
        super().__init__()
        
        self.parquet_path = parquet_path
        self.text_tokenizer = text_tokenizer
        self.pad_id = self.text_tokenizer.pad_token_id
        self.bos_id = showo_token_ids['bos_id']
        self.eos_id = showo_token_ids['eos_id']
        self.boi_id = showo_token_ids['boi_id']
        self.eoi_id = showo_token_ids['eoi_id']
        self.img_pad_id = showo_token_ids['img_pad_id']
        self.max_seq_len = max_seq_len
        self.image_size = image_size
        self.num_image_tokens = num_image_tokens
        self.h = latent_height
        self.w = latent_width
        self.cond_dropout_prob = cond_dropout_prob
        self.data_type = "interleaved_data"
        self.transform = image_transform
        self.max_num_images = max_num_images
        self.include_question_images = include_question_images
        self.include_solution_images = include_solution_images
        self.shuffle = shuffle
        
        # Get parquet file info
        self.pf = pq.ParquetFile(parquet_path)
        self.num_row_groups = self.pf.metadata.num_row_groups
        self.total_rows = self.pf.metadata.num_rows
        
        print(f"MathCanvas iterable dataset initialized. {self.total_rows} samples across {self.num_row_groups} row groups!")

        self.flag_tokens = self.text_tokenizer(
            "Mathematical reasoning with visual chain-of-thought.", add_special_tokens=False
        ).input_ids
        self.system_tokens = self.text_tokenizer(system, add_special_tokens=False).input_ids
        self.system_token_len = sum(len(tokens) for tokens in self.system_tokens)

        if len(self.system_tokens[0]) == 0:
            self.max_text_len = (
                                        max_seq_len
                                        - len(self.flag_tokens)
                                        - (num_image_tokens + 2) * max_num_images
                                        - 2
                                ) // max_num_images
        else:
            self.max_text_len = (
                                        max_seq_len
                                        - (num_image_tokens + 2) * max_num_images
                                        - 2
                                        - self.system_token_len
                                        - 1
                                ) // max_num_images

        self.min_res = min_res if min_res is not None else (256, 256)

    def _get_pil_image(self, image_data: Dict) -> Image.Image:
        """Convert image data from parquet to PIL Image."""
        if isinstance(image_data, dict) and 'bytes' in image_data:
            return Image.open(BytesIO(image_data['bytes'])).convert('RGB')
        elif isinstance(image_data, bytes):
            return Image.open(BytesIO(image_data)).convert('RGB')
        else:
            raise TypeError(f"Unsupported image data type: {type(image_data)}")

    def _process_row(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process a single row from the parquet file."""
        try:
            question_interleave = row.get('question_interleave', [])
            solution_interleave = row.get('solution_interleave', [])
            question_images = row.get('question_images', [])
            solution_images = row.get('solution_images', [])
            
            if len(question_interleave) == 0 and len(solution_interleave) == 0:
                return None

            image_list: List[Optional[torch.Tensor]] = []
            text_token_list: List[Optional[List[int]]] = []
            texts: List[str] = []
            total_images = 0

            # Process question part
            current_text = ""
            for item in question_interleave:
                if item['type'] == 'text':
                    content = item.get('content', '')
                    if content:
                        current_text += content + " "
                elif item['type'] == 'image' and self.include_question_images:
                    if total_images >= self.max_num_images:
                        continue
                    
                    if current_text.strip():
                        text_tokens = self.text_tokenizer(
                            current_text.strip(),
                            add_special_tokens=False,
                            truncation=True,
                            max_length=self.max_text_len,
                        ).input_ids
                        text_token_list.append(text_tokens)
                        texts.append(current_text.strip())
                        image_list.append(None)
                        current_text = ""
                    
                    img_idx = item.get('index', 0)
                    if img_idx < len(question_images):
                        try:
                            image = self._get_pil_image(question_images[img_idx])
                            image_tensor = self.transform(image, resolution=self.image_size)
                            image_list.append(image_tensor)
                            text_token_list.append(None)
                            texts.append('')
                            total_images += 1
                        except Exception:
                            continue

            if current_text.strip():
                current_text += "\n\nSolution: "
            else:
                current_text = "Solution: "

            # Process solution part
            for item in solution_interleave:
                if item['type'] == 'text':
                    content = item.get('content', '')
                    if content:
                        current_text += content + " "
                elif item['type'] == 'image' and self.include_solution_images:
                    if total_images >= self.max_num_images:
                        continue
                    
                    if current_text.strip():
                        text_tokens = self.text_tokenizer(
                            current_text.strip(),
                            add_special_tokens=False,
                            truncation=True,
                            max_length=self.max_text_len,
                        ).input_ids
                        text_token_list.append(text_tokens)
                        texts.append(current_text.strip())
                        current_text = ""
                    
                    img_idx = item.get('index', 0)
                    if img_idx < len(solution_images):
                        try:
                            image = self._get_pil_image(solution_images[img_idx])
                            image_tensor = self.transform(image, resolution=self.image_size)
                            image_list.append(image_tensor)
                            text_token_list.append(None)
                            texts.append('')
                            total_images += 1
                        except Exception:
                            continue

            if current_text.strip():
                answer = row.get('answer', '')
                if answer:
                    current_text += f"\n\nFinal Answer: {answer}"
                
                text_tokens = self.text_tokenizer(
                    current_text.strip(),
                    add_special_tokens=False,
                    truncation=True,
                    max_length=self.max_text_len,
                ).input_ids
                text_token_list.append(text_tokens)
                texts.append(current_text.strip())
                image_list.append(None)

            # Reorganize into pairs
            all_images = [img for img in image_list if img is not None]
            all_texts = [(t, txt) for t, txt in zip(text_token_list, texts) if t is not None]
            
            paired_images = []
            paired_text_tokens = []
            paired_texts = []
            
            max_pairs = max(len(all_images), len(all_texts))
            for i in range(max_pairs):
                if i < len(all_texts):
                    paired_text_tokens.append(all_texts[i][0])
                    paired_texts.append(all_texts[i][1])
                else:
                    paired_text_tokens.append([])
                    paired_texts.append('')
                
                if i < len(all_images):
                    paired_images.append(all_images[i])
                else:
                    paired_images.append(None)

            if len(paired_text_tokens) > 0 and paired_text_tokens[0] is not None:
                paired_text_tokens[0] = self.flag_tokens + paired_text_tokens[0]

            if len(paired_images) > self.max_num_images:
                paired_images = paired_images[:self.max_num_images]
                paired_text_tokens = paired_text_tokens[:self.max_num_images]
                paired_texts = paired_texts[:self.max_num_images]

            while len(paired_images) < self.max_num_images:
                paired_images.append(None)
                paired_text_tokens.append(None)
                paired_texts.append('')

            # Check valid content
            valid_content = any(t is not None and len(t) > 0 for t in paired_text_tokens) or \
                           any(img is not None for img in paired_images)
            if not valid_content:
                return None

            (
                text_tokens,
                text_labels,
                modality_positions,
                text_mask,
                image_mask,
            ) = format_interleaved_sequence(
                paired_images,
                paired_text_tokens,
                self.bos_id,
                self.eos_id,
                self.boi_id,
                self.eoi_id,
                self.pad_id,
                self.img_pad_id,
                self.num_image_tokens,
                self.max_seq_len,
                self.max_num_images,
            )

            text_labels[1: len(self.flag_tokens) + 1] = -100

            temp = []
            for img in paired_images:
                if img is not None:
                    temp.append(img)
                else:
                    temp.append(torch.zeros((3, self.image_size, self.image_size)))

            image = torch.stack(temp, dim=0)
            
            return {
                'text_tokens': text_tokens,
                'text_labels': text_labels,
                'images': image,
                'modality_positions': modality_positions,
                'text_masks': text_mask,
                'image_masks': image_mask,
                'texts': paired_texts,
                'data_type': self.data_type,
            }

        except Exception as e:
            print(f"Error processing row: {e}")
            return None

    def __iter__(self):
        """Iterate over the parquet file row groups."""
        worker_info = torch.utils.data.get_worker_info()
        
        if worker_info is None:
            # Single worker
            row_groups = list(range(self.num_row_groups))
        else:
            # Multiple workers - split row groups
            per_worker = self.num_row_groups // worker_info.num_workers
            worker_id = worker_info.id
            start = worker_id * per_worker
            end = start + per_worker if worker_id < worker_info.num_workers - 1 else self.num_row_groups
            row_groups = list(range(start, end))
        
        if self.shuffle:
            random.shuffle(row_groups)
        
        for rg_idx in row_groups:
            try:
                df = self.pf.read_row_group(rg_idx).to_pandas()
                indices = list(range(len(df)))
                if self.shuffle:
                    random.shuffle(indices)
                
                for idx in indices:
                    row = df.iloc[idx].to_dict()
                    result = self._process_row(row)
                    if result is not None:
                        yield result
            except Exception as e:
                print(f"Error reading row group {rg_idx}: {e}")
                continue

    def collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Collate function to batch data."""
        batched = collections.defaultdict(list)
        for data in batch:
            for key, value in data.items():
                batched[key].append(value)
        for key, value in batched.items():
            if key not in ('texts', 'data_type'):
                batched[key] = torch.stack(value, dim=0)
        return batched


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    from models.misc import get_text_tokenizer

    text_tokenizer, showo_token_ids = get_text_tokenizer(
        "Qwen/Qwen2.5-7B-Instruct",
        add_showo_tokens=True,
        return_showo_token_ids=True,
        llm_name="qwen2_5"
    )

    # Test with the MathCanvas dataset
    parquet_path = "/Users/denizakkoc/Desktop/master donem1/research/MathCanvas-Instruct-PlaneGeometry/plane_geometry.parquet"
    
    dataset = MathCanvasDataset(
        parquet_path=parquet_path,
        text_tokenizer=text_tokenizer,
        showo_token_ids=showo_token_ids,
        image_size=512,
        max_seq_len=5120,
        num_image_tokens=1024,
        latent_height=32,
        latent_width=32,
        max_num_images=4
    )
    
    train_dataloader = DataLoader(
        dataset, 
        batch_size=2, 
        collate_fn=dataset.collate_fn,
        shuffle=False, 
        num_workers=0
    )

    from tqdm import tqdm

    for i, data in tqdm(enumerate(train_dataloader)):
        print()
        print(data['data_type'], data['text_tokens'].shape, data['images'].shape)
        print(data['text_tokens'][0])
        print(data['modality_positions'][0])
        if i >= 2:
            break

