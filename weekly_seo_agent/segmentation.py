"""Compatibility wrapper. Use weekly_seo_agent.weekly_reporting_agent.segmentation."""
import weekly_seo_agent.weekly_reporting_agent.segmentation as _impl

# Re-export all non-dunder symbols (including private helpers used in tests).
globals().update({k: v for k, v in _impl.__dict__.items() if not k.startswith('__')})
