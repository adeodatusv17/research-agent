class DatasetAvailabilityChecker:
    def check(self, dataset_name: str) -> dict:
        return {"dataset_name": dataset_name, "available": False}
