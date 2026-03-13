import logging

def validate_columns(df, required_columns):

    logging.info("Validating dataset schema")

    missing = [col for col in required_columns if col not in df.columns]

    if missing:
        raise ValueError(
            f"Dataset missing required columns: {missing}"
        )
    
    logging.info("Dataset schema validation passed")