# 2024 인하 인공지능 챌린지

## 구조

```bash
.
├── ensemble # 앙상블 관련 config 파일입니다.
│   └── ensemble.yaml
├── model # train과 inference에 사용할 모델의 하이퍼파라미터입니다.
│   ├── best
│   │   ├── inference.yaml
│   │   └── train.yaml
│   ├── cos-dec
│   │   ├── inference.yaml
│   │   └── train.yaml
│   ├── cos-default
│   │   ├── inference.yaml
│   │   └── train.yaml
│   ├── cos-restart
│   │   ├── inference.yaml
│   │   └── train.yaml
│   └── ft-linear
│       ├── inference.yaml
│       └── train.yaml
├── setup.py
└── src # 소스코드입니다.
    ├── arguments.py
    ├── base.py
    ├── beam_generation.py
    ├── ensemble.py
    ├── eval.py
    ├── scheduler.py
    └── train.py
```

## 실행

Conda 혹은 Docker를 이용해 파일을 실행할 수 있습니다.

conda 

```bash
conda create -n [env_name] python=3.11
conda activate [env_name]
git clone https://github.com/jijihuny/ai_chat_qa_task
cd ai_chat_qa_task
bash entrypoint.sh [train | inference] # train과 inference 모드 중 선택할 수 있습니다.
```

docker

도커를 이용해 모델을 구동할 수 있습니다.
```bash
docker build . -t [docker_image_name]
docker run -it --rm --name [docker_container_name] -v ./:/workspace [docker_image_name] bash entrypoint.sh [train | inference]
```