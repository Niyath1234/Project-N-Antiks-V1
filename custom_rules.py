"""Dynamic rule management system for evolving prompts."""
import json
import os
from typing import List, Dict
from datetime import datetime


class CustomRuleManager:
	"""Manages user-defined rules that evolve the system prompt."""
	
	def __init__(self, rules_file: str = "custom_rules.json"):
		self.rules_file = rules_file
		self.rules: List[Dict] = []
		self.load_rules()
	
	def load_rules(self):
		"""Load rules from file."""
		if os.path.exists(self.rules_file):
			try:
				with open(self.rules_file, 'r', encoding='utf-8') as f:
					self.rules = json.load(f)
			except Exception:
				self.rules = []
		else:
			self.rules = []
	
	def save_rules(self):
		"""Save rules to file."""
		with open(self.rules_file, 'w', encoding='utf-8') as f:
			json.dump(self.rules, f, indent=2)
	
	def add_rule(self, rule_text: str, category: str = "general") -> int:
		"""Add a new rule. Returns the rule ID."""
		rule_id = len(self.rules) + 1
		rule = {
			"id": rule_id,
			"text": rule_text,
			"category": category,
			"created_at": datetime.now().isoformat(),
			"usage_count": 0
		}
		self.rules.append(rule)
		self.save_rules()
		return rule_id
	
	def remove_rule(self, rule_id: int) -> bool:
		"""Remove a rule by ID."""
		original_len = len(self.rules)
		self.rules = [r for r in self.rules if r["id"] != rule_id]
		if len(self.rules) < original_len:
			self.save_rules()
			return True
		return False
	
	def get_rule(self, rule_id: int) -> Dict:
		"""Get a specific rule by ID."""
		for rule in self.rules:
			if rule["id"] == rule_id:
				return rule
		return None
	
	def get_all_rules(self) -> List[Dict]:
		"""Get all rules."""
		return self.rules
	
	def get_rules_by_category(self, category: str) -> List[Dict]:
		"""Get rules by category."""
		return [r for r in self.rules if r["category"] == category]
	
	def increment_usage(self, rule_id: int):
		"""Increment usage count for a rule."""
		for rule in self.rules:
			if rule["id"] == rule_id:
				rule["usage_count"] = rule.get("usage_count", 0) + 1
				self.save_rules()
				break
	
	def format_rules_for_prompt(self) -> str:
		"""Format all rules as a prompt section."""
		if not self.rules:
			return ""
		
		lines = ["\n**CUSTOM USER-DEFINED RULES (MUST FOLLOW):**"]
		
		# Group by category
		categories = {}
		for rule in self.rules:
			cat = rule.get("category", "general")
			if cat not in categories:
				categories[cat] = []
			categories[cat].append(rule)
		
		for category, cat_rules in categories.items():
			lines.append(f"\n{category.upper()}:")
			for rule in cat_rules:
				lines.append(f"- [Rule #{rule['id']}] {rule['text']}")
				# Increment usage
				rule["usage_count"] = rule.get("usage_count", 0) + 1
		
		# Save updated usage counts
		self.save_rules()
		
		return "\n".join(lines)
	
	def format_rules_for_display(self) -> str:
		"""Format rules for display in CLI/UI."""
		if not self.rules:
			return "No custom rules defined yet."
		
		lines = ["📚 Custom Rules:\n"]
		for rule in self.rules:
			lines.append(f"#{rule['id']} [{rule['category']}] {rule['text']}")
			lines.append(f"   Created: {rule['created_at'][:10]}, Used: {rule.get('usage_count', 0)} times")
			lines.append("")
		
		return "\n".join(lines)
	
	def search_rules(self, query: str) -> List[Dict]:
		"""Search rules by text."""
		query_lower = query.lower()
		return [r for r in self.rules if query_lower in r["text"].lower()]
	
	def get_stats(self) -> Dict:
		"""Get statistics about rules."""
		total = len(self.rules)
		categories = {}
		total_usage = 0
		
		for rule in self.rules:
			cat = rule.get("category", "general")
			categories[cat] = categories.get(cat, 0) + 1
			total_usage += rule.get("usage_count", 0)
		
		return {
			"total_rules": total,
			"categories": categories,
			"total_usage": total_usage,
			"avg_usage_per_rule": total_usage / total if total > 0 else 0
		}


# Global instance
_rule_manager = None

def get_rule_manager(rules_file: str = None) -> CustomRuleManager:
	"""Get the global rule manager instance."""
	global _rule_manager
	if _rule_manager is None:
		if rules_file is None:
			rules_file = os.path.join(os.path.dirname(__file__), "custom_rules.json")
		_rule_manager = CustomRuleManager(rules_file)
	return _rule_manager

