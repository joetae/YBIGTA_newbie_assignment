#!/bin/bash

# miniconda가 존재하지 않을 경우 설치
## TODO
if ! command -v conda &> /dev/null; then
    echo "Miniconda가 설치되어 있지 않습니다. 설치를 시작합니다."
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
    bash miniconda.sh -b -p $HOME/miniconda
    rm miniconda.sh
    export PATH="$HOME/miniconda/bin:$PATH"
    source $HOME/miniconda/bin/activate
    conda init bash
    source ~/.bashrc
    echo "Miniconda 설치 완료."
else
    echo "Miniconda가 이미 설치되어 있습니다."
fi

# Conda 환경 생성 및 활성화
## TODO
ENV_NAME="myenv"
if ! conda env list | grep "$ENV_NAME" > /dev/null; then
    echo "가상 환경 '$ENV_NAME'가 없으므로 생성됩니다."
    conda create -n $ENV_NAME python=3.9 -y
fi

source $HOME/miniconda/bin/activate $ENV_NAME

## 건드리지 마세요! ##
python_env=$(python -c "import sys; print(sys.prefix)")
if [[ "$python_env" == *"/envs/myenv"* ]]; then
    echo "가상환경 활성화: 성공"
else
    echo "가상환경 활성화: 실패"
    exit 1 
fi

# 필요한 패키지 설치
## TODO
if ! command -v mypy &> /dev/null; then
    echo "mypy가 설치되어 있지 않습니다. 설치를 시작합니다."
    conda install -c conda-forge mypy || pip install mypy
    echo "mypy가 설치되었습니다."
fi

# Submission 폴더 파일 실행
cd submission || { echo "submission 디렉토리로 이동 실패"; exit 1; }

for file in *.py; do
    ## TODO
    input_file="../input/${file%.*}_input"
    output_file="../output/${file%.*}_output"
    python "$file" < "$input_file" > "$output_file"
    echo "$file 실행 완료"
done

# mypy 테스트 실행행
## TODO
echo "mypy를 통한 코드 타입 검사 중..."
mypy . || { echo "mypy 검사 실패"; exit 1; }
echo "mypy 검사 완료."

# 가상환경 비활성화
## TODO
echo "가상환경 비활성화중"
conda deactivate