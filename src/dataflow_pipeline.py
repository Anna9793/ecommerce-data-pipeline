import os
import json
import logging
from datetime import datetime
from typing import Dict, Any, Tuple, Iterable, Optional

import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions, StandardOptions, GoogleCloudOptions, SetupOptions
from apache_beam.transforms import window

logging.basicConfig(level=logging.INFO)

DEAD_LETTER_TAG = "dead_letter"

# ============================================================
# 1. Beam Transform DoFns with Dead-Letter Queue Routing
# ============================================================

class ParseTransactionDoFn(beam.DoFn):
    """
    Parses JSON transaction string/bytes into structured dictionary.
    Routes corrupted, malformed, or invalid records to the Dead-Letter Queue.
    """

    def process(self, element) -> Iterable[Any]:
        raw_str = ""
        try:
            if isinstance(element, bytes):
                raw_str = element.decode("utf-8")
            elif isinstance(element, str):
                raw_str = element
            else:
                raw_str = json.dumps(element)

            data = json.loads(raw_str)
            
            customer_id = data.get("CustomerID") or data.get("customer_id")
            if not customer_id or str(customer_id).strip() in ("", "None", "nan"):
                # Route missing Customer ID to Dead-Letter Queue
                yield beam.pvalue.TaggedOutput(
                    DEAD_LETTER_TAG,
                    {
                        "raw_payload": raw_str,
                        "error_type": "MISSING_CUSTOMER_ID",
                        "error_message": "Transaction rejected: missing or empty CustomerID",
                        "timestamp": datetime.utcnow().isoformat()
                    }
                )
                return
                
            qty = int(data.get("Quantity", 1))
            unit_price = float(data.get("UnitPrice", 0.0))
            
            if unit_price < 0:
                # Route negative unit price to Dead-Letter Queue
                yield beam.pvalue.TaggedOutput(
                    DEAD_LETTER_TAG,
                    {
                        "raw_payload": raw_str,
                        "error_type": "INVALID_UNIT_PRICE",
                        "error_message": f"Negative UnitPrice detected: {unit_price}",
                        "timestamp": datetime.utcnow().isoformat()
                    }
                )
                return

            inv_no = str(data.get("InvoiceNo", ""))
            is_cancel = inv_no.startswith("C") or qty < 0
            
            # Main valid output
            yield {
                "customer_id": str(customer_id),
                "invoice_no": inv_no,
                "stock_code": str(data.get("StockCode", "")),
                "description": str(data.get("Description", "")),
                "quantity": abs(qty),
                "unit_price": unit_price,
                "amount": round(abs(qty) * unit_price, 2),
                "is_cancel": is_cancel,
                "invoice_date": str(data.get("InvoiceDate", datetime.utcnow().isoformat()))
            }
        except Exception as e:
            # Route unparseable / syntax errors to Dead-Letter Queue
            logging.warning("Routing corrupted transaction to Dead-Letter Queue: %s", e)
            yield beam.pvalue.TaggedOutput(
                DEAD_LETTER_TAG,
                {
                    "raw_payload": str(element),
                    "error_type": "JSON_PARSE_ERROR",
                    "error_message": str(e),
                    "timestamp": datetime.utcnow().isoformat()
                }
            )

class AggregateCustomerMetrics(beam.CombineFn):
    """Accumulates transaction spend, count, and cancellations per window."""

    def create_accumulator(self) -> Dict[str, Any]:
        return {
            "total_spend": 0.0,
            "order_invoices": set(),
            "item_count": 0,
            "cancellation_count": 0
        }

    def add_input(self, accumulator: Dict[str, Any], input_element: Dict[str, Any]) -> Dict[str, Any]:
        accumulator["total_spend"] += input_element["amount"]
        accumulator["order_invoices"].add(input_element["invoice_no"])
        accumulator["item_count"] += input_element["quantity"]
        if input_element["is_cancel"]:
            accumulator["cancellation_count"] += 1
        return accumulator

    def merge_accumulators(self, accumulators: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
        merged = self.create_accumulator()
        for acc in accumulators:
            merged["total_spend"] += acc["total_spend"]
            merged["order_invoices"].update(acc["order_invoices"])
            merged["item_count"] += acc["item_count"]
            merged["cancellation_count"] += acc["cancellation_count"]
        return merged

    def extract_output(self, accumulator: Dict[str, Any]) -> Dict[str, Any]:
        order_count = len(accumulator["order_invoices"])
        cancellations = accumulator["cancellation_count"]
        return {
            "total_spend": round(accumulator["total_spend"], 2),
            "order_count": order_count,
            "item_count": accumulator["item_count"],
            "cancellation_count": cancellations,
            "cancellation_ratio": round(cancellations / max(order_count, 1), 4) if order_count > 0 else 0.0
        }

class FormatWindowedAggregateDoFn(beam.DoFn):
    """Enriches output with window boundaries and spending velocity."""

    def process(
        self,
        element: Tuple[str, Dict[str, Any]],
        window_param=beam.DoFn.WindowParam
    ) -> Iterable[Dict[str, Any]]:
        customer_id, metrics = element
        
        order_cnt = max(metrics["order_count"], 1)
        velocity = round(metrics["total_spend"] / order_cnt, 2)
        
        try:
            w_start = window_param.start.to_utc_datetime().isoformat()
            w_end = window_param.end.to_utc_datetime().isoformat()
        except Exception:
            w_start = datetime.utcnow().isoformat()
            w_end = datetime.utcnow().isoformat()

        yield {
            "customer_id": str(customer_id),
            "window_start": w_start,
            "window_end": w_end,
            "total_spend": float(metrics["total_spend"]),
            "order_count": int(metrics["order_count"]),
            "item_count": int(metrics["item_count"]),
            "cancellation_count": int(metrics["cancellation_count"]),
            "cancellation_ratio": float(metrics["cancellation_ratio"]),
            "spending_velocity": float(velocity)
        }

# ============================================================
# 2. Pipeline Builder Function
# ============================================================

def run_pipeline(
    project_id: str = "anna-ml-pipeline",
    pubsub_topic: str = "retail-transactions-topic",
    bq_dataset: str = "retail_data",
    bq_table: str = "streaming_customer_aggregates",
    window_size_seconds: int = 300,
    runner: str = "DirectRunner",
    pipeline_args: Optional[list] = None
):
    """Builds and executes the Apache Beam Dataflow streaming pipeline."""
    options = PipelineOptions(pipeline_args or [])
    options.view_as(StandardOptions).runner = runner
    
    if runner == "DataflowRunner":
        options.view_as(StandardOptions).streaming = True
        gcp_options = options.view_as(GoogleCloudOptions)
        gcp_options.project = project_id
        gcp_options.region = "us-central1"
        gcp_options.temp_location = f"gs://{project_id}-bucket/dataflow/temp"
        gcp_options.staging_location = f"gs://{project_id}-bucket/dataflow/staging"

    topic_path = f"projects/{project_id}/topics/{pubsub_topic}"
    table_spec = f"{project_id}:{bq_dataset}.{bq_table}"

    logging.info("Starting Apache Beam pipeline with %s on %s", runner, topic_path)

    with beam.Pipeline(options=options) as p:
        parsed_results = (
            p
            | "ReadFromPubSub" >> beam.io.ReadFromPubSub(topic=topic_path)
            | "ParseTransactionAndDLQ" >> beam.ParDo(ParseTransactionDoFn()).with_outputs(
                DEAD_LETTER_TAG,
                main="valid"
            )
        )

        # 1. Main Path: Valid transactions -> Windowing -> Aggregation -> BigQuery
        (
            parsed_results.valid
            | "FixedWindows" >> beam.WindowInto(window.FixedWindows(window_size_seconds))
            | "KeyByCustomer" >> beam.Map(lambda tx: (tx["customer_id"], tx))
            | "CombineCustomerMetrics" >> beam.CombinePerKey(AggregateCustomerMetrics())
            | "FormatWindowedAggregates" >> beam.ParDo(FormatWindowedAggregateDoFn())
            | "WriteToBigQuery" >> beam.io.WriteToBigQuery(
                table=table_spec,
                schema="SCHEMA_AUTODETECT",
                write_disposition=beam.io.BigQueryDisposition.WRITE_APPEND,
                create_disposition=beam.io.BigQueryDisposition.CREATE_NEVER
            )
        )

        # 2. Dead-Letter Path: Log and monitor corrupted records
        (
            parsed_results[DEAD_LETTER_TAG]
            | "FormatDLQLogs" >> beam.Map(lambda err: logging.error("DLQ EVENT: %s", json.dumps(err)))
        )

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run Apache Beam Dataflow streaming pipeline.")
    parser.add_argument("--runner", default="DirectRunner", choices=["DirectRunner", "DataflowRunner"])
    parser.add_argument("--window_size", type=int, default=300)
    args, beam_args = parser.parse_known_args()

    project = os.getenv("GCP_PROJECT", "anna-ml-pipeline")
    run_pipeline(
        project_id=project,
        window_size_seconds=args.window_size,
        runner=args.runner,
        pipeline_args=beam_args
    )
