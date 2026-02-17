"""Compatibility wrapper. Use weekly_seo_agent.weekly_reporting_agent.additional_context."""
import weekly_seo_agent.weekly_reporting_agent.additional_context as _impl

# Re-export all non-dunder symbols (including private helpers used in tests).
globals().update({k: v for k, v in _impl.__dict__.items() if not k.startswith('__')})
