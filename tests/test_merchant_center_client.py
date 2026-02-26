from weekly_seo_agent.weekly_reporting_agent.clients.merchant_center_client import (
    MerchantCenterClient,
)


def test_validate_country_access_ok_with_mca_subaccounts(monkeypatch):
    monkeypatch.setattr(
        MerchantCenterClient, "_read_service_account_email", lambda self: "svc@example.com"
    )
    client = MerchantCenterClient(
        credentials_path="secret.json",
        mca_id="9000",
        account_id_map={"PL": "1001", "CZ": "2002"},
    )
    monkeypatch.setattr(
        client, "_fetch_accessible_account_ids", lambda: {"9000", "1001"}
    )
    monkeypatch.setattr(client, "_fetch_subaccount_ids", lambda _: {"2002", "3003"})

    result = client.validate_country_access(["PL", "CZ"])
    assert result["ok"] is True
    assert result["missing_country_mappings"] == []
    assert result["inaccessible_country_accounts"] == {}
    assert result["validated_country_accounts"]["PL"] == "1001"
    assert result["validated_country_accounts"]["CZ"] == "2002"


def test_validate_country_access_detects_missing_mapping(monkeypatch):
    monkeypatch.setattr(
        MerchantCenterClient, "_read_service_account_email", lambda self: "svc@example.com"
    )
    client = MerchantCenterClient(
        credentials_path="secret.json",
        mca_id="",
        account_id_map={"PL": "1001"},
    )
    monkeypatch.setattr(client, "_fetch_accessible_account_ids", lambda: {"1001"})
    monkeypatch.setattr(client, "_fetch_subaccount_ids", lambda _: set())

    result = client.validate_country_access(["PL", "CZ"])
    assert result["ok"] is False
    assert result["missing_country_mappings"] == ["CZ"]
    assert result["inaccessible_country_accounts"] == {}


def test_validate_country_access_detects_inaccessible_ids(monkeypatch):
    monkeypatch.setattr(
        MerchantCenterClient, "_read_service_account_email", lambda self: "svc@example.com"
    )
    client = MerchantCenterClient(
        credentials_path="secret.json",
        mca_id="",
        account_id_map={"PL": "1001", "CZ": "2002"},
    )
    monkeypatch.setattr(client, "_fetch_accessible_account_ids", lambda: {"1001"})
    monkeypatch.setattr(client, "_fetch_subaccount_ids", lambda _: set())

    result = client.validate_country_access(["PL", "CZ"])
    assert result["ok"] is False
    assert result["missing_country_mappings"] == []
    assert result["inaccessible_country_accounts"] == {"CZ": "2002"}
