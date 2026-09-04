import json
import pytest
beam = pytest.importorskip("apache_beam")
from apache_beam.testing.test_pipeline import TestPipeline
from apache_beam.testing.util import assert_that, equal_to
from src.dataflow_pipeline import (
    ParseTransactionDoFn,
    AggregateCustomerMetrics,
    FormatWindowedAggregateDoFn,
    DEAD_LETTER_TAG
)

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

def test_parse_transaction_routes_missing_customer_to_dlq():
    dofn = ParseTransactionDoFn()
    raw_json = json.dumps({
        "InvoiceNo": "581494",
        "Quantity": 1,
        "UnitPrice": 10.0,
        "CustomerID": ""
    })

    results = list(dofn.process(raw_json))
    assert len(results) == 1
    tagged_out = results[0]
    assert isinstance(tagged_out, beam.pvalue.TaggedOutput)
    assert tagged_out.tag == DEAD_LETTER_TAG
    assert tagged_out.value["error_type"] == "MISSING_CUSTOMER_ID"

def test_parse_transaction_routes_negative_price_to_dlq():
    dofn = ParseTransactionDoFn()
    raw_json = json.dumps({
        "InvoiceNo": "581495",
        "Quantity": 1,
        "UnitPrice": -15.0,
        "CustomerID": "17850"
    })

    results = list(dofn.process(raw_json))
    assert len(results) == 1
    tagged_out = results[0]
    assert isinstance(tagged_out, beam.pvalue.TaggedOutput)
    assert tagged_out.tag == DEAD_LETTER_TAG
    assert tagged_out.value["error_type"] == "INVALID_UNIT_PRICE"

def test_parse_transaction_routes_corrupted_json_to_dlq():
    dofn = ParseTransactionDoFn()
    corrupted_data = "THIS_IS_NOT_VALID_JSON{foo:"

    results = list(dofn.process(corrupted_data))
    assert len(results) == 1
    tagged_out = results[0]
    assert isinstance(tagged_out, beam.pvalue.TaggedOutput)
    assert tagged_out.tag == DEAD_LETTER_TAG
    assert tagged_out.value["error_type"] == "JSON_PARSE_ERROR"

def test_beam_pipeline_with_dlq_branching():
    test_events = [
        json.dumps({"InvoiceNo": "INV1", "CustomerID": "101", "Quantity": 2, "UnitPrice": 10.0}),
        json.dumps({"InvoiceNo": "INV2", "CustomerID": "", "Quantity": 1, "UnitPrice": 20.0}), # Corrupt
        json.dumps({"InvoiceNo": "INV3", "CustomerID": "101", "Quantity": 1, "UnitPrice": 30.0}),
    ]

    with TestPipeline() as p:
        results = (
            p
            | "CreateData" >> beam.Create(test_events)
            | "ParseAndDLQ" >> beam.ParDo(ParseTransactionDoFn()).with_outputs(
                DEAD_LETTER_TAG,
                main="valid"
            )
        )

        aggregated = (
            results.valid
            | "KeyByCust" >> beam.Map(lambda tx: (tx["customer_id"], tx))
            | "Combine" >> beam.CombinePerKey(AggregateCustomerMetrics())
        )

        expected_valid = [
            ("101", {"total_spend": 50.0, "order_count": 2, "item_count": 3, "cancellation_count": 0, "cancellation_ratio": 0.0})
        ]

        assert_that(aggregated, equal_to(expected_valid), label="CheckValidBranch")
