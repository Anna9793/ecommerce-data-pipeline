import json
import pytest
import apache_beam as beam
from apache_beam.testing.test_pipeline import TestPipeline
from apache_beam.testing.util import assert_that, equal_to
from src.dataflow_pipeline import ParseTransactionDoFn, AggregateCustomerMetrics, FormatWindowedAggregateDoFn

def test_parse_transaction_dofn_valid():
    dofn = ParseTransactionDoFn()
    raw_json = json.dumps({
        "InvoiceNo": "581492",
        "StockCode": "85123A",
        "Description": "WHITE HANGING HEART T-LIGHT HOLDER",
        "Quantity": 2,
        "UnitPrice": 2.55,
        "CustomerID": "17850"
    })

    results = list(dofn.process(raw_json))
    assert len(results) == 1
    tx = results[0]
    assert tx["customer_id"] == "17850"
    assert tx["amount"] == 5.10
    assert tx["is_cancel"] is False

def test_parse_transaction_dofn_cancellation():
    dofn = ParseTransactionDoFn()
    raw_json = json.dumps({
        "InvoiceNo": "C581493",
        "StockCode": "85123A",
        "Description": "CANCELLATION",
        "Quantity": -2,
        "UnitPrice": 2.55,
        "CustomerID": "17850"
    })

    results = list(dofn.process(raw_json))
    assert len(results) == 1
    tx = results[0]
    assert tx["is_cancel"] is True
    assert tx["quantity"] == 2

def test_parse_transaction_dofn_skips_anonymous():
    dofn = ParseTransactionDoFn()
    raw_json = json.dumps({
        "InvoiceNo": "581494",
        "Quantity": 1,
        "UnitPrice": 10.0,
        "CustomerID": ""
    })

    results = list(dofn.process(raw_json))
    assert len(results) == 0

def test_beam_pipeline_customer_aggregations():
    test_events = [
        json.dumps({"InvoiceNo": "INV1", "CustomerID": "101", "Quantity": 2, "UnitPrice": 10.0}),
        json.dumps({"InvoiceNo": "INV2", "CustomerID": "101", "Quantity": 1, "UnitPrice": 30.0}),
        json.dumps({"InvoiceNo": "C_INV3", "CustomerID": "101", "Quantity": -1, "UnitPrice": 10.0}),
        json.dumps({"InvoiceNo": "INV4", "CustomerID": "102", "Quantity": 5, "UnitPrice": 4.0}),
    ]

    with TestPipeline() as p:
        output = (
            p
            | "CreateData" >> beam.Create(test_events)
            | "Parse" >> beam.ParDo(ParseTransactionDoFn())
            | "KeyByCust" >> beam.Map(lambda tx: (tx["customer_id"], tx))
            | "Combine" >> beam.CombinePerKey(AggregateCustomerMetrics())
        )

        expected = [
            ("101", {"total_spend": 60.0, "order_count": 3, "item_count": 4, "cancellation_count": 1, "cancellation_ratio": 0.3333}),
            ("102", {"total_spend": 20.0, "order_count": 1, "item_count": 5, "cancellation_count": 0, "cancellation_ratio": 0.0})
        ]

        assert_that(output, equal_to(expected))
