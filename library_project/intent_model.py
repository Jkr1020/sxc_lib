import json
import os
from pathlib import Path


class IntentClassifier:
    """
    Simple Intent Classifier for Library AI Chatbot.
    Classifies user queries into predefined intents based on keyword matching.
    """
    
    def __init__(self, intent_data_path="intent_data.json"):
        """
        Initialize the Intent Classifier.
        
        Args:
            intent_data_path: Path to intent training data (optional)
        """
        self.intent_data_path = intent_data_path
        self.intents = self._initialize_intents()
    
    def _initialize_intents(self):
        """
        Initialize intent patterns. If intent_data.json exists, load from there.
        Otherwise, use default patterns.
        """
        # Try to load from JSON file
        if os.path.exists(self.intent_data_path):
            try:
                with open(self.intent_data_path, 'r') as f:
                    return json.load(f)
            except Exception:
                pass
        
        # Default intent patterns
        return {
            "greeting": {
                "patterns": ["hi", "hello", "hey", "greetings", "good morning", "good afternoon", "good evening"],
                "priority": 5
            },
            "library_info": {
                "patterns": ["library", "rules", "timing", "hours", "fine", "principal", "history", "contact", "information", "about"],
                "priority": 4
            },
            "student_status": {
                "patterns": ["my books", "my status", "pending", "issue", "return", "due", "borrowed", "books with me"],
                "priority": 3
            },
            "book_search": {
                "patterns": ["search", "find", "book", "title", "author", "look for", "where is"],
                "priority": 2
            },
            "recommendation": {
                "patterns": ["recommend", "suggest", "best", "popular", "trending", "new", "what should i read"],
                "priority": 1
            }
        }
    
    def predict(self, message):
        """
        Predict the intent of a user message based on keyword matching.
        
        Args:
            message: User input message
            
        Returns:
            Intent label (string)
        """
        if not message or not isinstance(message, str):
            return "default"
        
        message_lower = message.lower().strip()
        
        # Find matching intents
        matches = {}
        for intent_name, intent_data in self.intents.items():
            patterns = intent_data.get("patterns", [])
            priority = intent_data.get("priority", 0)
            
            # Check if any pattern matches
            for pattern in patterns:
                if pattern in message_lower:
                    matches[intent_name] = priority
                    break
        
        # Return the intent with highest priority
        if matches:
            return max(matches, key=matches.get)
        
        # Default fallback
        return "default"
