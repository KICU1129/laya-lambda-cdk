"""API Gateway HTTP API → LambdaコンテナによるLaya推論。"""
from __future__ import annotations

import json
from pathlib import Path
from aws_cdk import CfnOutput, Duration, RemovalPolicy, Size, Stack, Tags
from aws_cdk import aws_apigatewayv2 as apigwv2
from aws_cdk import aws_apigatewayv2_authorizers as authorizers
from aws_cdk import aws_apigatewayv2_integrations as integrations
from aws_cdk import aws_ecr_assets as ecr_assets
from aws_cdk import aws_iam as iam
from aws_cdk import aws_lambda as lambda_
from aws_cdk import aws_logs as logs
from constructs import Construct
from .config import DeploymentConfig


class LayaApiStack(Stack):
    def __init__(self, scope: Construct, construct_id: str, *,
                 config: DeploymentConfig | None = None, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)
        cfg = config or DeploymentConfig()
        asset_dir = Path(__file__).resolve().parents[1] / "lambda"

        function_logs = logs.LogGroup(self, "FunctionLogs",
            retention=logs.RetentionDays.ONE_WEEK,
            removal_policy=RemovalPolicy.DESTROY)
        api_logs = logs.LogGroup(self, "ApiAccessLogs",
            retention=logs.RetentionDays.ONE_WEEK,
            removal_policy=RemovalPolicy.DESTROY)
        role = iam.Role(self, "InferenceRole",
            assumed_by=iam.ServicePrincipal("lambda.amazonaws.com"))
        # S3、EFS、Bedrock、Hugging Face認証情報などの権限は不要。
        function_logs.grant_write(role)

        self.function = lambda_.DockerImageFunction(self, "InferenceFunction",
            code=lambda_.DockerImageCode.from_image_asset(
                str(asset_dir), platform=ecr_assets.Platform.LINUX_AMD64,
                exclude=["__pycache__", "**/__pycache__", "*.pyc", "**/*.pyc"]),
            architecture=lambda_.Architecture.X86_64,
            memory_size=cfg.memory_mb,
            timeout=Duration.seconds(28),
            ephemeral_storage_size=Size.mebibytes(512),
            reserved_concurrent_executions=cfg.reserved_concurrency,
            role=role,
            log_group=function_logs,
            environment={
                "MODEL_DIR": "/opt/laya-model",
                "HF_HOME": "/tmp/huggingface",
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "HF_HUB_DISABLE_TELEMETRY": "1",
                "DO_NOT_TRACK": "1",
                "TOKENIZERS_PARALLELISM": "false",
                "TORCH_NUM_THREADS": str(cfg.torch_threads),
                "OMP_NUM_THREADS": str(cfg.torch_threads),
                "MKL_NUM_THREADS": str(cfg.torch_threads),
                "LAYA_MAX_LEN": str(cfg.max_len),
                "LAYA_HEAD_MAX_LEN": str(cfg.head_max_len),
                "LAYA_MAX_QUESTIONS": str(cfg.max_questions),
                "LOG_LEVEL": "INFO",
            },
            description="Offline CPU inference; one baked-in multilingual Laya model")

        # APIは必ずaliasを呼ぶ。PCを$LATESTに設定する構成にはしない。
        version = self.function.current_version
        version.apply_removal_policy(RemovalPolicy.DESTROY)
        self.alias = lambda_.Alias(self, "LiveAlias", alias_name="live",
            version=version,
            provisioned_concurrent_executions=cfg.provisioned_concurrency or None)

        self.api = apigwv2.HttpApi(self, "HttpApi",
            api_name=f"{self.stack_name}-laya",
            create_default_stage=False,
            default_authorizer=authorizers.HttpIamAuthorizer())
        integration = integrations.HttpLambdaIntegration("InferenceIntegration", self.alias,
            payload_format_version=apigwv2.PayloadFormatVersion.VERSION_2_0,
            timeout=Duration.seconds(29))
        self.api.add_routes(path="/predict", methods=[apigwv2.HttpMethod.POST], integration=integration)
        self.api.add_routes(path="/health", methods=[apigwv2.HttpMethod.GET], integration=integration)

        # L1を使い、アクセスログ/スロットルを明示的に設定する。
        apigwv2.CfnStage(self, "DefaultStage", api_id=self.api.api_id,
            stage_name="$default", auto_deploy=True,
            access_log_settings=apigwv2.CfnStage.AccessLogSettingsProperty(
                destination_arn=api_logs.log_group_arn,
                format=json.dumps({
                    "requestId": "$context.requestId",
                    "routeKey": "$context.routeKey",
                    "status": "$context.status",
                    "responseLength": "$context.responseLength",
                    "integrationLatency": "$context.integrationLatency",
                }, separators=(",", ":"))),
            default_route_settings=apigwv2.CfnStage.RouteSettingsProperty(
                throttling_rate_limit=2, throttling_burst_limit=4,
                detailed_metrics_enabled=False))

        CfnOutput(self, "ApiUrl", value=self.api.api_endpoint)
        CfnOutput(self, "PredictUrl", value=f"{self.api.api_endpoint}/predict")
        CfnOutput(self, "HealthUrl", value=f"{self.api.api_endpoint}/health")
        CfnOutput(self, "FunctionName", value=self.function.function_name)
        CfnOutput(self, "FunctionAliasArn", value=self.alias.function_arn)
        CfnOutput(self, "FunctionLogGroup", value=function_logs.log_group_name)
        CfnOutput(self, "ApiLogGroup", value=api_logs.log_group_name)
        CfnOutput(self, "AwsRegion", value=self.region)
        CfnOutput(self, "PredictInvokeArn", value=(
            f"arn:{self.partition}:execute-api:{self.region}:{self.account}:"
            f"{self.api.api_id}/$default/POST/predict"))
        CfnOutput(self, "HealthInvokeArn", value=(
            f"arn:{self.partition}:execute-api:{self.region}:{self.account}:"
            f"{self.api.api_id}/$default/GET/health"))
        CfnOutput(self, "ProvisionedConcurrency", value=str(cfg.provisioned_concurrency))
        Tags.of(self).add("Project", "laya-lambda-api")
