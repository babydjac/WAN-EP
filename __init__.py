"""
WAN 2.2 Expert Prompter - ComfyUI Custom Node Package

Professional WAN 2.2 video generation prompt optimization using Gemini AI.
Transforms vague user concepts into expert-level 80-120 word structured prompts
following WAN 2.2 MoE architecture requirements and cinematographic best practices.

Author: Custom Node Generator
Version: 1.0.0
"""

from .wan_prompter import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

# Export the main node mappings for ComfyUI discovery
__all__ = [
    "NODE_CLASS_MAPPINGS", 
    "NODE_DISPLAY_NAME_MAPPINGS"
]

# Optional: Add version info
__version__ = "1.0.0"
__description__ = "WAN 2.2 Expert Prompter using Gemini AI"

# Expose web extensions directory so ComfyUI serves and loads it
# This includes both the original color script and the new enhanced UI
WEB_DIRECTORY = "./web"
