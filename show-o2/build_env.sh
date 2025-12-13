# Remove conflicting Colab packages
pip3 uninstall -y jax jaxlib 2>/dev/null || true;

pip3 install torch==2.5.1;
pip3 install transformers==4.47.0;
pip3 install diffusers==0.31.0;
pip3 install einops==0.8.0;
pip3 install decord;
pip3 install gpustat;
pip3 install sentencepiece;
pip3 install ipdb;
pip3 install ftfy regex tqdm;
pip3 install git+https://github.com/openai/CLIP.git;
pip3 install onnx==1.17.0;
pip3 install onnxsim;
pip3 install omegaconf;
pip3 install torchdiffeq;
pip3 install segment_anything;
pip3 install lightning==2.4.0;
pip3 install wandb;
# Install pre-built flash-attn wheel (much faster than building from source)
pip3 install flash-attn --no-build-isolation || pip3 install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.3/flash_attn-2.7.3+cu12torch2.5cxx11abiFALSE-cp312-cp312-linux_x86_64.whl;
pip3 install deepspeed==0.15.3;
# Updated accelerate and huggingface-hub for peft compatibility
pip3 install accelerate>=0.26.0;
pip3 install timm==1.0.12;
pip3 install huggingface-hub>=0.25.0;
pip3 install peft==0.13.0;
pip3 install onnxruntime==1.20.1;
pip3 install dill;
pip3 install pandas;
pip3 install pyarrow;
pip3 install av==12.0.0;
pip3 install moviepy;
pip3 install tensorflow==2.16.1;
pip3 install decord;
pip3 install jsonlines;