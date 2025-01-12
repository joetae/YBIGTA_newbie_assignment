#!/bin/bash

# miniconda가 존재하지 않을 경우 설치
## TODO
if ! command -v conda &> /dev/null; then
    echo "miniconda 설치중"
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
    bash miniconda.sh -b -p "$HOME/miniconda"
    export PATH="$HOME/miniconda/bin:$PATH"
    conda init
    source ~/.bashrc
else
    echo "miniconda가 이미 존재합니다"
fi

# Conda 환경 생성 및 활성화
## TODO
if ! conda env list | grep -q myenv; then
    echo "가상환경 생성중"
    conda create -y -n myenv python=3.9
fi
echo "가상환경 활성화중"
source ~/miniconda/bin/activate myenv

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
echo "mypy 설치중"
pip install mypy

# Submission 폴더 파일 실행
cd submission || { echo "submission 디렉토리로 이동 실패"; exit 1; }

for file in *.py; do
    ## TODO
    if ! head -n 1 "$file" | grep -q "coding: utf-8"; then
        sed -i '1s/^/# -*- coding: utf-8 -*-\n/' "$file"
    fi
    
    iconv -f euc-kr -t utf-8 "$file" -o "$file" 2>/dev/null || echo "$file 변환 실패"

    base_name=$(basename "$file" .py)
    input_file="../input/${base_name}_input"
    output_file="../output/${base_name}_output"
    python "$file" < "$input_file" > "$output_file"
done

# mypy 테스트 실행
## TODO
echo "mypy 테스트 실행중"
mypy *.py

# 가상환경 비활성화
## TODO
echo "가상환경 비활성화중"
conda deactivate