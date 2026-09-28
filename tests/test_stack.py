"""CDKをインストールした環境ではCloudFormationテンプレートを合成して検査する。"""
import pytest
cdk = pytest.importorskip("aws_cdk", reason="CDK is not installed in this environment")
from aws_cdk.assertions import Template, Match
from infra.config import DeploymentConfig
from infra.stack import LayaApiStack


def template(config=None):
    app = cdk.App()
    stack = LayaApiStack(app, "TestLaya", config=config,
                        env=cdk.Environment(account="111111111111", region="ap-northeast-1"))
    return Template.from_stack(stack)


def test_private_invocation_and_image_lambda():
    t = template()
    t.has_resource_properties("AWS::Lambda::Function", {
        "PackageType":"Image", "Architectures":["x86_64"], "MemorySize":8192,
        "Timeout":28, "ReservedConcurrentExecutions":2,
        "Environment":{"Variables":Match.object_like({"HF_HUB_OFFLINE":"1", "MODEL_DIR":"/opt/laya-model"})}})
    t.has_resource_properties("AWS::ApiGatewayV2::Api", {"ProtocolType":"HTTP"})
    t.has_resource_properties("AWS::ApiGatewayV2::Route", {"RouteKey":"POST /predict", "AuthorizationType":"AWS_IAM"})
    t.has_resource_properties("AWS::ApiGatewayV2::Route", {"RouteKey":"GET /health", "AuthorizationType":"AWS_IAM"})
    t.resource_count_is("AWS::Lambda::Url", 0)
    t.has_resource_properties("AWS::ApiGatewayV2::Integration", {
        "PayloadFormatVersion":"2.0", "TimeoutInMillis":29000})
    t.has_resource_properties("AWS::ApiGatewayV2::Stage", {
        "StageName":"$default", "AutoDeploy":True,
        "DefaultRouteSettings":Match.object_like({"ThrottlingRateLimit":2,"ThrottlingBurstLimit":4})})
    t.has_resource_properties("AWS::Lambda::Alias", {"Name":"live", "ProvisionedConcurrencyConfig":Match.absent()})


def test_provisioned_concurrency_is_on_alias():
    t = template(DeploymentConfig(provisioned_concurrency=1))
    t.has_resource_properties("AWS::Lambda::Alias", {
        "Name":"live", "ProvisionedConcurrencyConfig":{"ProvisionedConcurrentExecutions":1}})
