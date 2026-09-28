# 上流ソフトウェアとモデル

このソース配布ZIPはモデル重みや依存ライブラリのバイナリを含みません。Dockerビルド時に取得します。

| 名称 | 用途 | 上流 |
|---|---|---|
| Laya 0.3.21 | 判断モデルの推論ランタイム | https://github.com/NandhaKishorM/laya |
| laya-multilingual | 日本語を含む多言語判断モデル | https://huggingface.co/convaiinnovations/laya-multilingual |
| PyTorch CPU | CPU推論 | https://pytorch.org/ |
| Transformers | エンコーダーとtokenizerのロード | https://github.com/huggingface/transformers |
| Hugging Face Hub | ビルド中のモデル取得 | https://github.com/huggingface/huggingface_hub |
| Safetensors / NumPy | 重みロード・数値処理 | https://github.com/huggingface/safetensors / https://numpy.org/ |
| AWS CDK / AWS Lambda Python base image | インフラ定義とランタイム | https://github.com/aws/aws-cdk / https://gallery.ecr.aws/lambda/python |

確認したLayaリポジトリとモデルカードのライセンス表示はApache-2.0です。モデルのREADMEと、存在するLICENSE/NOTICEファイルはモデルの取得時に同梱します。その他の上流ライセンスは各配布物に従ってください。ベースイメージのOSパッケージや推移的な依存関係も別のライセンスを持ちます。

本プロジェクトはLayaまたはAWSの公式配布物ではありません。依存バージョンの更新時は互換性とライセンス表示を再確認してください。
