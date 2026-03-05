from src.ingestion import run_ingestion

def main():
    print("Starting data pipeline...")
    run_ingestion()
    print("Pipeline finished.")

if __name__ == "__main__":
    main()