"""Laya推論APIのCDKエントリーポイント。"""
from __future__ import annotations

import os
import aws_cdk as cdk
from infra.config import DeploymentConfig
from infra.stack import LayaApiStack

app = cdk.App()
config = DeploymentConfig.from_context(app.node.try_get_context)
LayaApiStack(
    app,
    "LayaApiStack",
    config=config,
    env=cdk.Environment(
        account=os.environ.get("CDK_DEFAULT_ACCOUNT"),
        region=os.environ.get("CDK_DEFAULT_REGION"),
    ),
    description="IAM-authorized HTTP API and CPU Laya inference on Lambda",
)
app.synth()
