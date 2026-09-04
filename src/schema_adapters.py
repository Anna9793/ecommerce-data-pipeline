import json
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, Any, Optional, Union
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)

# ============================================================
# 1. Canonical Data Contract Model
# ============================================================

class CanonicalTransaction(BaseModel):
    """Universal internal transaction model used across all pipeline components."""
    tenant_id: str = Field(default="giftshop_uk", description="Tenant / Store Identifier")
    customer_id: str = Field(..., description="Unique customer identifier or email")
    invoice_no: str = Field(..., description="Order / Invoice identifier")
    stock_code: str = Field(default="GENERAL", description="Item SKU or Stock Code")
    description: str = Field(default="", description="Product title / line item description")
    quantity: int = Field(default=1, description="Number of items purchased")
    unit_price: float = Field(default=0.0, description="Price per unit in EUR")
    amount: float = Field(default=0.0, description="Total line item amount in EUR")
    is_cancel: bool = Field(default=False, description="Whether this transaction was cancelled/refunded")
    invoice_date: str = Field(default_factory=lambda: datetime.utcnow().isoformat(), description="ISO Timestamp")

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump() if hasattr(self, "model_dump") else self.dict()

    def to_bigquery_row(self) -> Dict[str, Any]:
        """Maps canonical fields to the BigQuery retail_data.transactions table schema."""
        country_map = {
            "giftshop_uk": "United Kingdom",
            "nordic_tech": "Sweden",
            "shopify": "Sweden",
            "olist": "Brazil",
            "olist_marketplace": "Brazil"
        }
        return {
            "InvoiceNo": str(self.invoice_no),
            "StockCode": str(self.stock_code),
            "Description": str(self.description),
            "Quantity": -abs(self.quantity) if self.is_cancel else abs(self.quantity),
            "InvoiceDate": str(self.invoice_date),
            "UnitPrice": float(self.unit_price),
            "CustomerID": str(self.customer_id).strip(),
            "Country": country_map.get(self.tenant_id, "United Kingdom")
        }

# ============================================================
# 2. Base Adapter Interface
# ============================================================

class BaseSchemaAdapter(ABC):
    """Abstract Base Class for external schema normalization adapters."""

    @abstractmethod
    def to_canonical(self, raw_data: Dict[str, Any], tenant_id: Optional[str] = None) -> CanonicalTransaction:
        """Translates raw source payload into CanonicalTransaction."""
        pass

# ============================================================
# 3. Concrete Adapters
# ============================================================

class UciRetailAdapter(BaseSchemaAdapter):
    """Adapter for baseline UCI Online Retail schema (InvoiceNo, CustomerID, UnitPrice)."""

    def to_canonical(self, raw_data: Dict[str, Any], tenant_id: Optional[str] = None) -> CanonicalTransaction:
        cust_id = str(raw_data.get("CustomerID") or raw_data.get("customer_id") or "GUEST").strip()
        inv_no = str(raw_data.get("InvoiceNo") or raw_data.get("invoice_no") or "").strip()
        qty = int(raw_data.get("Quantity", 1))
        unit_price = float(raw_data.get("UnitPrice", 0.0))
        
        is_cancel = inv_no.startswith("C") or qty < 0
        total_amount = round(abs(qty) * unit_price, 2)
        date_str = str(raw_data.get("InvoiceDate") or datetime.utcnow().isoformat())

        return CanonicalTransaction(
            tenant_id=tenant_id or "giftshop_uk",
            customer_id=cust_id,
            invoice_no=inv_no,
            stock_code=str(raw_data.get("StockCode", "GENERAL")),
            description=str(raw_data.get("Description", "")),
            quantity=abs(qty),
            unit_price=unit_price,
            amount=total_amount,
            is_cancel=is_cancel,
            invoice_date=date_str
        )

class ShopifyAdapter(BaseSchemaAdapter):
    """Adapter for Shopify Orders export & webhook schema (Name, Email, Lineitem price, Financial Status)."""

    def to_canonical(self, raw_data: Dict[str, Any], tenant_id: Optional[str] = None) -> CanonicalTransaction:
        cust_id = str(raw_data.get("Customer ID") or raw_data.get("Email") or raw_data.get("email") or "GUEST").strip()
        inv_no = str(raw_data.get("Name") or raw_data.get("order_id") or raw_data.get("id") or "").strip()
        
        status = str(raw_data.get("Financial Status") or raw_data.get("financial_status") or "paid").lower()
        is_cancel = status in ("refunded", "voided", "cancelled")
        
        qty = int(raw_data.get("Lineitem quantity") or raw_data.get("quantity") or 1)
        unit_price = float(raw_data.get("Lineitem price") or raw_data.get("price") or 0.0)
        total_amount = round(abs(qty) * unit_price, 2)
        date_str = str(raw_data.get("Created at") or raw_data.get("created_at") or datetime.utcnow().isoformat())

        return CanonicalTransaction(
            tenant_id=tenant_id or "nordic_tech",
            customer_id=cust_id,
            invoice_no=inv_no,
            stock_code=str(raw_data.get("Lineitem sku") or raw_data.get("sku") or "SKU-SHOPIFY"),
            description=str(raw_data.get("Lineitem name") or raw_data.get("title") or "Shopify Product"),
            quantity=abs(qty),
            unit_price=unit_price,
            amount=total_amount,
            is_cancel=is_cancel,
            invoice_date=date_str
        )

class OlistAdapter(BaseSchemaAdapter):
    """Adapter for Brazilian Olist Marketplace schema (order_id, customer_id, price, order_status)."""

    def to_canonical(self, raw_data: Dict[str, Any], tenant_id: Optional[str] = None) -> CanonicalTransaction:
        cust_id = str(raw_data.get("customer_id") or "GUEST").strip()
        inv_no = str(raw_data.get("order_id") or "").strip()
        status = str(raw_data.get("order_status") or "delivered").lower()
        is_cancel = status in ("canceled", "unavailable")
        
        qty = int(raw_data.get("order_item_id") or raw_data.get("quantity") or 1)
        unit_price = float(raw_data.get("price") or 0.0)
        total_amount = round(abs(qty) * unit_price, 2)
        date_str = str(raw_data.get("order_purchase_timestamp") or datetime.utcnow().isoformat())

        return CanonicalTransaction(
            tenant_id=tenant_id or "olist_marketplace",
            customer_id=cust_id,
            invoice_no=inv_no,
            stock_code=str(raw_data.get("product_id") or "PROD-OLIST"),
            description=str(raw_data.get("product_category_name") or "Marketplace Item"),
            quantity=abs(qty),
            unit_price=unit_price,
            amount=total_amount,
            is_cancel=is_cancel,
            invoice_date=date_str
        )

# ============================================================
# 4. Schema Adapter Factory
# ============================================================

class SchemaAdapterFactory:
    """Factory for selecting and applying the appropriate schema adapter."""

    _adapters: Dict[str, BaseSchemaAdapter] = {
        "uci": UciRetailAdapter(),
        "giftshop_uk": UciRetailAdapter(),
        "shopify": ShopifyAdapter(),
        "nordic_tech": ShopifyAdapter(),
        "olist": OlistAdapter(),
        "olist_marketplace": OlistAdapter()
    }

    @classmethod
    def register_adapter(cls, source_name: str, adapter: BaseSchemaAdapter):
        """Registers a new external adapter dynamically (Open-Closed Principle)."""
        cls._adapters[source_name.lower()] = adapter
        logging.info("Registered new schema adapter for source: %s", source_name)

    @classmethod
    def get_adapter(cls, source_or_tenant: Optional[str] = None) -> BaseSchemaAdapter:
        """Retrieves adapter by source name or tenant ID, defaulting to UciRetailAdapter."""
        if source_or_tenant and source_or_tenant.lower() in cls._adapters:
            return cls._adapters[source_or_tenant.lower()]
        return cls._adapters["uci"]

    @classmethod
    def normalize(cls, raw_data: Union[str, bytes, Dict[str, Any]], source_or_tenant: Optional[str] = None) -> CanonicalTransaction:
        """Auto-detects format or uses explicit source to return CanonicalTransaction."""
        if isinstance(raw_data, bytes):
            raw_data = raw_data.decode("utf-8")
        if isinstance(raw_data, str):
            data_dict = json.loads(raw_data)
        else:
            data_dict = raw_data

        # Auto-detect if source is unspecified
        if not source_or_tenant:
            if "Lineitem price" in data_dict or "Financial Status" in data_dict or "Email" in data_dict:
                source_or_tenant = "shopify"
            elif "order_purchase_timestamp" in data_dict or "order_id" in data_dict and "price" in data_dict:
                source_or_tenant = "olist"
            else:
                source_or_tenant = "uci"

        adapter = cls.get_adapter(source_or_tenant)
        return adapter.to_canonical(data_dict, tenant_id=source_or_tenant)
