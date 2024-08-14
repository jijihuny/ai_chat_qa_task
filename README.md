# 2024 인하 인공지능 챌린지

conda 

```bash
conda create -n [env_name] -f environment.yaml
conda activate [env_name]
git clone https://github.com/jijihuny/ai_chat_qa_task
bash entrypoint.sh [train | inference]
```
<!-- 
docker

```bash
docker build . -t [docker_image_name]
docker run [docker_image_name] bash entrypoint.sh [train | inference]
``` -->