import json
import pytest
from src.schema_adapters import (
    CanonicalTransaction,
    UciRetailAdapter,
    ShopifyAdapter,
    OlistAdapter,
    SchemaAdapterFactory,
    BaseSchemaAdapter
)

def test_uci_retail_adapter():
    raw = {
        "InvoiceNo": "581492",
        "StockCode": "85123A",
        "Description": "WHITE HANGING HEART T-LIGHT HOLDER",
        "Quantity": 3,
        "UnitPrice": 2.55,
        "CustomerID": "17850",
        "InvoiceDate": "2026-01-10T12:00:00"
    }

    adapter = UciRetailAdapter()
    tx = adapter.to_canonical(raw)

    assert isinstance(tx, CanonicalTransaction)
    assert tx.customer_id == "17850"
    assert tx.invoice_no == "581492"
    assert tx.unit_price == 2.55
    assert tx.amount == 7.65
    assert tx.is_cancel is False

def test_shopify_adapter_paid_order():
    raw_shopify = {
        "Name": "#1001",
        "Email": "erik@nordic.se",
        "Financial Status": "paid",
        "Lineitem quantity": 2,
        "Lineitem price": 149.00,
        "Lineitem name": "Nordic Pro ANC Wireless Headphones",
        "Lineitem sku": "SKU-TECH-001",
        "Created at": "2026-03-01T09:15:22Z"
    }

    adapter = ShopifyAdapter()
    tx = adapter.to_canonical(raw_shopify, tenant_id="nordic_tech")

    assert isinstance(tx, CanonicalTransaction)
    assert tx.tenant_id == "nordic_tech"
    assert tx.customer_id == "erik@nordic.se"
    assert tx.invoice_no == "#1001"
    assert tx.amount == 298.00
    assert tx.is_cancel is False
    assert tx.description == "Nordic Pro ANC Wireless Headphones"

def test_shopify_adapter_refunded_order():
    raw_shopify_refund = {
        "Name": "#1004",
        "Email": "freja@nordic.se",
        "Financial Status": "refunded",
        "Lineitem quantity": 1,
        "Lineitem price": 45.00,
        "Lineitem name": "Ergonomic Gaming Mouse"
    }

    adapter = ShopifyAdapter()
    tx = adapter.to_canonical(raw_shopify_refund)

    assert tx.is_cancel is True
    assert tx.amount == 45.00

def test_olist_adapter():
    raw_olist = {
        "order_id": "e481f51cbdc50e4be2176ec5cf329221",
        "customer_id": "9ef432eb625a2f9c7e",
        "price": 29.99,
        "order_item_id": 1,
        "order_status": "delivered",
        "product_category_name": "utilidades_domesticas"
    }

    adapter = OlistAdapter()
    tx = adapter.to_canonical(raw_olist)

    assert tx.customer_id == "9ef432eb625a2f9c7e"
    assert tx.invoice_no == "e481f51cbdc50e4be2176ec5cf329221"
    assert tx.amount == 29.99
    assert tx.is_cancel is False

def test_factory_auto_detection():
    # 1. Auto-detect Shopify payload
    shopify_json = json.dumps({
        "Name": "#1005",
        "Email": "magnus@nordic.se",
        "Lineitem price": 29.99,
        "Lineitem quantity": 1
    })
    tx1 = SchemaAdapterFactory.normalize(shopify_json)
    assert tx1.tenant_id == "shopify" or tx1.tenant_id == "nordic_tech"
    assert tx1.customer_id == "magnus@nordic.se"

    # 2. Auto-detect UCI payload
    uci_json = json.dumps({
        "InvoiceNo": "123456",
        "CustomerID": "19999",
        "UnitPrice": 10.0,
        "Quantity": 1
    })
    tx2 = SchemaAdapterFactory.normalize(uci_json)
    assert tx2.customer_id == "19999"

def test_factory_dynamic_registration():
    class CustomWooCommerceAdapter(BaseSchemaAdapter):
        def to_canonical(self, raw_data, tenant_id=None):
            return CanonicalTransaction(
                tenant_id=tenant_id or "woocommerce_store",
                customer_id=str(raw_data["billing_email"]),
                invoice_no=str(raw_data["order_key"]),
                amount=float(raw_data["total"]),
                quantity=1
            )

    SchemaAdapterFactory.register_adapter("woocommerce", CustomWooCommerceAdapter())
    
    woo_data = {"billing_email": "test@woo.com", "order_key": "WOO-99", "total": 50.0}
    tx = SchemaAdapterFactory.normalize(woo_data, source_or_tenant="woocommerce")
    assert tx.customer_id == "test@woo.com"
    assert tx.invoice_no == "WOO-99"
    assert tx.amount == 50.0

def test_to_bigquery_row_mapping():
    shopify_json = {
        "Name": "#1008",
        "Email": "astrid@nordic.se",
        "Lineitem price": 89.50,
        "Lineitem quantity": 2,
        "Lineitem name": "Wool Sweater",
        "Lineitem sku": "SKU-SWEATER"
    }
    tx = SchemaAdapterFactory.normalize(shopify_json)
    bq_row = tx.to_bigquery_row()

    assert "InvoiceNo" in bq_row
    assert "StockCode" in bq_row
    assert "UnitPrice" in bq_row
    assert "CustomerID" in bq_row
    assert "Country" in bq_row
    assert isinstance(bq_row["CustomerID"], str)
    assert bq_row["CustomerID"] == "astrid@nordic.se"
    assert bq_row["InvoiceNo"] == "#1008"
    assert bq_row["UnitPrice"] == 89.50
    assert bq_row["StockCode"] == "SKU-SWEATER"

    # Also test numeric string ID preservation
    uci_tx = CanonicalTransaction(customer_id="17850", invoice_no="581492", amount=10.0, quantity=1, unit_price=10.0)
    uci_bq = uci_tx.to_bigquery_row()
    assert uci_bq["CustomerID"] == "17850"
