# Green AI Tensor-Decomposition Experiment

CIFAR-10 pretrained ResNet 계열 모델을 기준으로 Tucker-2/HOSVD 기반 저랭크 분해를 적용해 원본, 50% 목표 압축, 75% 목표 압축을 비교합니다.

GitHub Actions에서 자동으로:
- CIFAR-10 다운로드
- pretrained baseline 로드
- Tucker-2 압축
- 압축 모델 fine-tuning
- 정확도, 실제 파라미터 비율, CPU latency 측정
- Linux RAPL이 노출될 경우 J/inference 측정
- results.csv / report.md artifact 업로드

GitHub-hosted runner에서 RAPL이 노출되지 않으면 에너지는 N/A로 남기며 임의 추정하지 않습니다.
