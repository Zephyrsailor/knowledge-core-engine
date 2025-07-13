"""Golden dataset management for evaluation."""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Iterator
from datetime import datetime

from .evaluator import TestCase

logger = logging.getLogger(__name__)


class GoldenDataset:
    """Manages golden test datasets for evaluation."""
    
    def __init__(self, name: str = "default"):
        """Initialize golden dataset.
        
        Args:
            name: Name of the dataset
        """
        self.name = name
        self.test_cases: List[TestCase] = []
        self.metadata: Dict[str, Any] = {
            "name": name,
            "created_at": datetime.now().isoformat(),
            "version": "1.0.0"
        }
        self._index: Dict[str, int] = {}  # Map test_case_id to index
    
    def add_test_case(self, question: str, ground_truth: str, 
                      contexts: List[str], metadata: Optional[Dict[str, Any]] = None,
                      test_case_id: Optional[str] = None) -> TestCase:
        """Add a test case to the dataset.
        
        Args:
            question: The question
            ground_truth: Expected answer
            contexts: Relevant contexts
            metadata: Optional metadata
            test_case_id: Optional ID
            
        Returns:
            Created test case
        """
        test_case = TestCase(
            question=question,
            ground_truth=ground_truth,
            contexts=contexts,
            metadata=metadata or {},
            test_case_id=test_case_id
        )
        
        self.test_cases.append(test_case)
        self._index[test_case.test_case_id] = len(self.test_cases) - 1
        
        logger.debug(f"Added test case {test_case.test_case_id} to dataset {self.name}")
        return test_case
    
    def add_from_dict(self, data: Dict[str, Any]) -> TestCase:
        """Add test case from dictionary.
        
        Args:
            data: Dictionary with test case data
            
        Returns:
            Created test case
        """
        return self.add_test_case(
            question=data["question"],
            ground_truth=data["ground_truth"],
            contexts=data["contexts"],
            metadata=data.get("metadata", {}),
            test_case_id=data.get("test_case_id")
        )
    
    def get_by_id(self, test_case_id: str) -> Optional[TestCase]:
        """Get test case by ID.
        
        Args:
            test_case_id: Test case ID
            
        Returns:
            Test case or None if not found
        """
        index = self._index.get(test_case_id)
        if index is not None:
            return self.test_cases[index]
        return None
    
    def filter_by(self, **kwargs) -> List[TestCase]:
        """Filter test cases by metadata.
        
        Args:
            **kwargs: Metadata key-value pairs to filter by
            
        Returns:
            Filtered test cases
        """
        filtered = []
        
        for test_case in self.test_cases:
            match = True
            for key, value in kwargs.items():
                if test_case.metadata.get(key) != value:
                    match = False
                    break
            
            if match:
                filtered.append(test_case)
        
        return filtered
    
    def sample(self, n: int, random_state: Optional[int] = None) -> List[TestCase]:
        """Random sample of test cases.
        
        Args:
            n: Number of samples
            random_state: Random seed for reproducibility
            
        Returns:
            Sampled test cases
        """
        import random
        
        if random_state is not None:
            random.seed(random_state)
        
        n = min(n, len(self.test_cases))
        return random.sample(self.test_cases, n)
    
    def split(self, test_ratio: float = 0.2, 
              random_state: Optional[int] = None) -> tuple['GoldenDataset', 'GoldenDataset']:
        """Split dataset into train and test sets.
        
        Args:
            test_ratio: Ratio of test set
            random_state: Random seed
            
        Returns:
            Train dataset, test dataset
        """
        import random
        
        if random_state is not None:
            random.seed(random_state)
        
        # Shuffle indices
        indices = list(range(len(self.test_cases)))
        random.shuffle(indices)
        
        # Split
        test_size = int(len(indices) * test_ratio)
        test_indices = set(indices[:test_size])
        
        # Create datasets
        train_dataset = GoldenDataset(f"{self.name}_train")
        test_dataset = GoldenDataset(f"{self.name}_test")
        
        for i, test_case in enumerate(self.test_cases):
            if i in test_indices:
                test_dataset.test_cases.append(test_case)
            else:
                train_dataset.test_cases.append(test_case)
        
        # Rebuild indices
        train_dataset._rebuild_index()
        test_dataset._rebuild_index()
        
        return train_dataset, test_dataset
    
    def _rebuild_index(self):
        """Rebuild the index mapping."""
        self._index = {}
        for i, test_case in enumerate(self.test_cases):
            self._index[test_case.test_case_id] = i
    
    def save(self, file_path: Path) -> None:
        """Save dataset to file.
        
        Args:
            file_path: Path to save file
        """
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "metadata": self.metadata,
            "test_cases": [
                {
                    "test_case_id": tc.test_case_id,
                    "question": tc.question,
                    "ground_truth": tc.ground_truth,
                    "contexts": tc.contexts,
                    "metadata": tc.metadata,
                    "generated_answer": tc.generated_answer
                }
                for tc in self.test_cases
            ]
        }
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Saved dataset {self.name} with {len(self.test_cases)} test cases to {file_path}")
    
    @classmethod
    def load(cls, file_path: Path) -> 'GoldenDataset':
        """Load dataset from file.
        
        Args:
            file_path: Path to load from
            
        Returns:
            Loaded dataset
        """
        file_path = Path(file_path)
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Create dataset
        dataset = cls(data["metadata"].get("name", "loaded"))
        dataset.metadata = data["metadata"]
        
        # Load test cases
        for tc_data in data["test_cases"]:
            test_case = TestCase(
                question=tc_data["question"],
                ground_truth=tc_data["ground_truth"],
                contexts=tc_data["contexts"],
                metadata=tc_data.get("metadata", {}),
                test_case_id=tc_data.get("test_case_id"),
                generated_answer=tc_data.get("generated_answer")
            )
            dataset.test_cases.append(test_case)
            dataset._index[test_case.test_case_id] = len(dataset.test_cases) - 1
        
        logger.info(f"Loaded dataset {dataset.name} with {len(dataset.test_cases)} test cases")
        return dataset
    
    def to_dataframe(self):
        """Convert to pandas DataFrame.
        
        Returns:
            DataFrame with test cases
        """
        try:
            import pandas as pd
            
            data = []
            for tc in self.test_cases:
                row = {
                    "test_case_id": tc.test_case_id,
                    "question": tc.question,
                    "ground_truth": tc.ground_truth,
                    "num_contexts": len(tc.contexts),
                    "has_generated_answer": tc.generated_answer is not None
                }
                # Add metadata fields
                row.update(tc.metadata)
                data.append(row)
            
            return pd.DataFrame(data)
        
        except ImportError:
            logger.warning("pandas not installed, cannot convert to DataFrame")
            return None
    
    def statistics(self) -> Dict[str, Any]:
        """Get dataset statistics.
        
        Returns:
            Dictionary with statistics
        """
        stats = {
            "total_cases": len(self.test_cases),
            "avg_contexts_per_case": sum(len(tc.contexts) for tc in self.test_cases) / len(self.test_cases) if self.test_cases else 0,
            "avg_question_length": sum(len(tc.question) for tc in self.test_cases) / len(self.test_cases) if self.test_cases else 0,
            "avg_answer_length": sum(len(tc.ground_truth) for tc in self.test_cases) / len(self.test_cases) if self.test_cases else 0,
            "categories": {}
        }
        
        # Count by category
        for tc in self.test_cases:
            category = tc.metadata.get("category", "uncategorized")
            stats["categories"][category] = stats["categories"].get(category, 0) + 1
        
        return stats
    
    def __len__(self) -> int:
        """Get number of test cases."""
        return len(self.test_cases)
    
    def __getitem__(self, index: int) -> TestCase:
        """Get test case by index."""
        return self.test_cases[index]
    
    def __iter__(self) -> Iterator[TestCase]:
        """Iterate over test cases."""
        return iter(self.test_cases)
    
    def __repr__(self) -> str:
        """String representation."""
        return f"GoldenDataset(name='{self.name}', size={len(self.test_cases)})"