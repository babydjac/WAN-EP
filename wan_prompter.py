"""
WAN 2.2 Expert Prompter Node
Creates optimized WAN 2.2 video generation prompts from vague user inputs using Gemini API.

Based on WAN 2.2 Master Prompter Training Manual specifications:
- 80-120 word optimal prompt structure
- Sequential structure: Opening Scene → Camera Motion → Reveal/Payoff
- Professional cinematographic vocabulary
- Mixture-of-Experts architecture optimization
"""

import json
import re
from typing import Optional, Tuple, List, Dict
import torch
from enum import Enum

from comfy.comfy_types.node_typing import IO, ComfyNodeABC, InputTypeDict
from server import PromptServer

# Import Gemini API components (assuming they're available in ComfyUI environment)
try:
    from comfy_api_nodes.apis import (
        GeminiContent,
        GeminiGenerateContentRequest,
        GeminiGenerateContentResponse,
        GeminiPart,
    )
    from comfy_api_nodes.apis.client import (
        ApiEndpoint,
        HttpMethod,
        SynchronousOperation,
    )
    from comfy_api_nodes.apinode_utils import validate_string
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False


class WAN22Model(str, Enum):
    """WAN 2.2 Model variants with their characteristics"""
    T2V_A14B = "T2V-A14B"
    I2V_A14B = "I2V-A14B"
    TI2V_5B = "TI2V-5B"


class WANCinematicStyle(str, Enum):
    """Cinematic style presets for WAN 2.2"""
    DRAMATIC = "dramatic"
    GENTLE = "gentle"
    CINEMATIC = "cinematic"
    REALISTIC = "realistic"
    STYLIZED = "stylized"
    COMMERCIAL = "commercial"


class WANSceneType(str, Enum):
    """Scene type categories for optimal camera work"""
    STATIC_SCENE = "static_scene"
    CHARACTER_FOCUS = "character_focus"
    ACTION_SEQUENCE = "action_sequence"
    ATMOSPHERIC = "atmospheric"


class WAN22ExpertPrompter(ComfyNodeABC):
    @classmethod
    def INPUT_TYPES(cls) -> InputTypeDict:
        gemini_models = [
            "gemini-2.5-pro-preview-05-06",
            "gemini-2.5-flash-preview-04-17"
        ] if GEMINI_AVAILABLE else ["Gemini API not available"]

        return {
            "required": {
                "user_subject": (
                    IO.STRING,
                    {
                        "multiline": True,
                        "default": "A warrior in a moonlit forest",
                        "tooltip": "Describe your video concept in simple terms.",
                    },
                ),
                "wan_model": (
                    IO.COMBO,
                    {
                        "options": [model.value for model in WAN22Model],
                        "default": WAN22Model.T2V_A14B.value,
                    },
                ),
                "cinematic_style": (
                    IO.COMBO,
                    {
                        "options": [style.value for style in WANCinematicStyle],
                        "default": WANCinematicStyle.CINEMATIC.value,
                    },
                ),
                "scene_type": (
                    IO.COMBO,
                    {
                        "options": [scene.value for scene in WANSceneType],
                        "default": WANSceneType.CHARACTER_FOCUS.value,
                    },
                ),
                "gemini_model": (
                    IO.COMBO,
                    {
                        "options": gemini_models,
                        "default": gemini_models[0],
                    },
                ),
            },
            "optional": {
                "reference_image": (
                    IO.IMAGE,
                    {"default": None},
                ),
                "custom_instructions": (
                    IO.STRING,
                    {"multiline": True, "default": ""},
                ),
                "nsfw_mode": (
                    IO.COMBO,
                    {
                        "options": [False, True],
                        "default": False,
                        "tooltip": "Enable explicit NSFW prompting",
                    },
                ),
            },
            "hidden": {
                "auth_token": "AUTH_TOKEN_COMFY_ORG",
                "comfy_api_key": "API_KEY_COMFY_ORG",
                "unique_id": "UNIQUE_ID",
            },
        }

    DESCRIPTION = "🎬 WAN 2.2 Expert Prompter powered by Gemini AI with enhanced UI."
    RETURN_TYPES = ("STRING", "STRING", "STRING", "INT", "STRING")
    RETURN_NAMES = ("optimized_prompt", "negative_prompt", "technical_analysis", "word_count", "mode_status")
    FUNCTION = "generate_expert_prompt"
    CATEGORY = "video/WAN 2.2"
    API_NODE = True

    def get_wan_expert_system_prompt(self, wan_model: str, style: str, scene_type: str, nsfw_mode: bool = False) -> str:
        import random
        
        # Rotate different creative approaches to avoid repetition
        creative_approaches = [
            "You are a visionary cinematographer with the soul of a poet and the eye of a master painter.",
            "You are a legendary film director known for creating iconic, unforgettable visual moments.", 
            "You are a cinematic alchemist who transforms simple concepts into pure visual gold.",
            "You are a visual storyteller who speaks fluent in the language of light, shadow, and motion.",
            "You are a master of cinematic emotion who finds the extraordinary in every frame."
        ]
        
        opening_approach = random.choice(creative_approaches)
        base_prompt = f"""{opening_approach}

You are creating WAN 2.2 video generation prompts with uncompromising technical standards and limitless creative vision.

CORE WAN 2.2 RULES (NON-NEGOTIABLE):
- EXACTLY 80-120 words total
- Structure: Opening Scene → Camera Motion → Reveal/Payoff
- Professional cinematographic vocabulary mandatory

TARGET CONFIGURATION:
- Model: {wan_model}
- Style: {style} 
- Scene Type: {scene_type}

CREATIVE MANDATE:
- AVOID formulaic or template-like language
- Create UNIQUE, imaginative scenarios every time
- Use unexpected visual metaphors and creative descriptions
- Blend multiple cinematic techniques organically
- Make each prompt feel like a mini film treatment

RICH VOCABULARY PALETTE:
Camera Movement: gliding crane sweep, intimate dolly push, vertiginous tilt, spiraling orbit, whip pan, rack focus pull, floating steadicam flow
Lighting Artistry: chiaroscuro contrast, dappled sunlight, neon-soaked shadows, ethereal backlighting, harsh tungsten spill, soft window light, practical source motivation
Visual Poetry: gossamer mist, crystalline reflections, velvet darkness, liquid mercury textures, painterly bokeh, silk-smooth motion blur
Composition: rule of thirds tension, leading lines convergence, negative space breathing, foreground layering, depth compression

STYLE-SPECIFIC INSPIRATION:
- Dramatic: Think Blade Runner 2049, Mad Max Fury Road - high contrast, bold movements
- Cinematic: Channel Terrence Malick, Roger Deakins - poetic, flowing, naturalistic  
- Commercial: Apple ads, luxury brand films - clean, aspirational, precise
- Realistic: Documentary style, handheld intimacy, natural lighting
- Stylized: Wes Anderson symmetry, Denis Villeneuve scale, Wong Kar-wai color"""
        if nsfw_mode:
            base_prompt += """
NSFW MODE:
- Use explicit sexual terminology (penis, vagina, breasts, nipples, ass, cum, oral sex, vaginal sex, anal sex, etc.)
- Describe literal sexual acts clearly (penetration, licking, sucking, ejaculation, etc.)
- Maintain cinematographic framing and vocabulary
"""

        base_prompt += f"""

SCENE-SPECIFIC CREATIVE DIRECTION:
- Static Scene: Focus on atmospheric details, subtle lighting shifts, micro-movements that build tension
- Character Focus: Intimate camera work, emotional beats, facial expression reveals, personal space invasion
- Action Sequence: Dynamic movement vocabulary, kinetic energy, impact moments, rhythm and pacing
- Atmospheric: Environmental storytelling, mood through lighting/weather, symbolic elements

CREATIVE PROCESS:
1. Read the user's concept and find the emotional CORE
2. Imagine this as a pivotal scene in a legendary film
3. Choose unexpected but fitting visual metaphors
4. Weave cinematography that serves the story emotion
5. End with a reveal that recontextualizes everything

OUTPUT REQUIREMENTS:
- ONE creative prompt: exactly 80-120 words
- Structure: Opening Scene → Camera Motion → Reveal/Payoff  
- NO formulaic language - make it feel organic and cinematic
- Include brief technical analysis of your choices

INSPIRATION EXAMPLES (do NOT copy - use as creative springboard):
"Weathered hands trace ancient symbols on crumbling stone. Camera drifts through shafts of cathedral light, golden motes dancing like memories made visible. The lens pulls focus from calloused fingertips to reveal towering cavern walls covered in identical markings. A slow orbit reveals the figure is not alone - dozens of robed silhouettes emerge from shadow, their synchronized breathing creating ethereal fog that catches the amber light, transforming the sacred space into a living, breathing temple of forgotten knowledge."
"""
        return base_prompt

    def enhance_user_subject(self, user_subject: str, style: str, scene_type: str) -> str:
        """Enhance the user's basic subject with creative context and emotional depth"""
        import random
        
        # Creative enhancement templates based on style and scene type
        enhancement_templates = {
            'dramatic': {
                'character_focus': [
                    "Focus on the raw emotional intensity and inner conflict of {subject}. What hidden truth is about to be revealed?",
                    "Capture the moment of transformation for {subject}. How does their world shift in this pivotal scene?",
                    "Show {subject} at a crossroads of destiny. What choice will define their fate?"
                ],
                'action_sequence': [
                    "Create explosive kinetic energy around {subject}. What force drives this unstoppable momentum?",
                    "Design a high-stakes confrontation involving {subject}. What are they fighting to protect or destroy?",
                    "Build relentless tension with {subject} at the center. What happens when all control is lost?"
                ],
                'static_scene': [
                    "Find the hidden drama in the stillness of {subject}. What tension lies beneath the surface?",
                    "Create atmospheric pressure around {subject}. What is waiting to explode into action?",
                    "Show {subject} in a moment of calm before the storm. What forces are gathering?"
                ],
                'atmospheric': [
                    "Immerse {subject} in an environment that reflects their inner state. What does the world reveal about them?",
                    "Use weather, lighting, and space to tell {subject}'s story. How does the atmosphere become a character?",
                    "Create a symbolic landscape around {subject}. What does their environment prophesy?"
                ]
            },
            'cinematic': {
                'character_focus': [
                    "Frame {subject} with poetic intimacy, finding beauty in vulnerability. What universal truth do they embody?",
                    "Capture {subject} in a moment of quiet revelation. How does light reveal their soul?",
                    "Show the delicate humanity of {subject}. What makes this moment eternally beautiful?"
                ],
                'atmospheric': [
                    "Blend {subject} organically with their natural environment. How do they become one with the world?",
                    "Create a dreamlike quality around {subject}. What memories or emotions float through this space?",
                    "Find the poetry in everyday moments with {subject}. How does the ordinary become extraordinary?"
                ]
            },
            'commercial': {
                'character_focus': [
                    "Present {subject} as the embodiment of aspiration and success. What lifestyle do they represent?",
                    "Show {subject} in their element of mastery and confidence. What excellence do they demonstrate?",
                    "Capture {subject} living their best life. What dream are they fulfilling?"
                ]
            },
            'realistic': {
                'character_focus': [
                    "Show {subject} with documentary authenticity. What real human truth are we witnessing?",
                    "Capture the unguarded moment with {subject}. What genuine emotion breaks through?",
                    "Present {subject} without artifice or pretense. What makes them completely human?"
                ]
            },
            'stylized': {
                'character_focus': [
                    "Frame {subject} with geometric precision and visual symmetry. What aesthetic perfection do they inhabit?",
                    "Place {subject} in a world of heightened visual reality. What stylistic universe contains them?",
                    "Show {subject} as part of a carefully curated visual language. What design philosophy governs their world?"
                ]
            }
        }
        
        # Get appropriate templates for this style and scene type
        style_templates = enhancement_templates.get(style, enhancement_templates['cinematic'])
        scene_templates = style_templates.get(scene_type, style_templates.get('character_focus', [
            "Transform {subject} into a cinematic moment. What story wants to be told?",
            "Find the visual poetry in {subject}. How does this become unforgettable?",
            "Elevate {subject} to legendary status. What makes this scene iconic?"
        ]))
        
        # Select a random enhancement template
        template = random.choice(scene_templates)
        
        # Create the enhanced subject
        enhanced = f"""Original concept: {user_subject}

Creative Direction: {template.format(subject=user_subject)}

Remember: This should feel like a pivotal scene from a masterpiece film. What emotional truth are we revealing through pure visual storytelling?"""
        
        return enhanced

    def validate_prompt_structure(self, prompt: str) -> Dict[str, any]:
        words = len(prompt.split())
        return {
            'word_count_valid': 80 <= words <= 120,
            'word_count': words,
            'has_camera_movement': any(t in prompt.lower() for t in ['pan','tilt','dolly','track','zoom','orbit','crane','camera']),
            'has_visual_elements': any(t in prompt.lower() for t in ['light','shadow','bokeh','focus','color']),
            'has_sequence_structure': any(t in prompt.lower() for t in ['→','then','as','while','revealing']),
            'overall_score': 0  # simplified
        }

    def create_gemini_parts(self, system_prompt: str, user_input: str, reference_image: Optional[torch.Tensor] = None) -> List:
        parts = [GeminiPart(text=f"{system_prompt}\n\nUser Subject: {user_input}")]
        if reference_image is not None:
            from comfy_api_nodes.apinode_utils import tensor_to_base64_string
            from comfy_api_nodes.apis import GeminiInlineData, GeminiMimeType
            image_b64 = tensor_to_base64_string(reference_image[0].unsqueeze(0))
            parts.append(GeminiPart(inlineData=GeminiInlineData(mimeType=GeminiMimeType.image_png, data=image_b64)))
        return parts

    def extract_prompt_components(self, response_text: str) -> Tuple[str, str, str]:
        lines = response_text.strip().split('\n')
        buffer, negative_prompt, analysis = [], "", ""
        section = "prompt"

        for line in lines:
            line = line.strip()
            if not line:
                continue
            if "negative" in line.lower() and "prompt" in line.lower():
                section = "negative"; continue
            elif "analysis" in line.lower() or "technical" in line.lower():
                section = "analysis"; continue
            if "prompt:" in line.lower() and len(line.split()) < 8:
                continue  # skip headers like "Here is the professional WAN 2.2 prompt:"

            if section == "prompt":
                buffer.append(line)
            elif section == "negative":
                negative_prompt += " " + line
            elif section == "analysis":
                analysis += " " + line

        optimized_prompt = ""
        if buffer:
            optimized_prompt = max(buffer, key=lambda x: len(x.split())).strip()

        if not optimized_prompt:
            for line in lines:
                if len(line.split()) > 20:
                    optimized_prompt = line.strip()
                    break

        return (
            optimized_prompt or "Failed to generate optimized prompt",
            negative_prompt.strip() or "–no text overlay, –no distorted faces, –no blurriness",
            analysis.strip() or "Generated using WAN 2.2 expert prompting methodology"
        )

    async def generate_expert_prompt(
        self,
        user_subject: str,
        wan_model: str,
        cinematic_style: str,
        scene_type: str,
        gemini_model: str,
        reference_image: Optional[torch.Tensor] = None,
        custom_instructions: str = "",
        nsfw_mode: bool = False,
        unique_id: Optional[str] = None,
        **kwargs
    ) -> Tuple[str, str, str, int, str]:
        if not GEMINI_AVAILABLE:
            mode_status = "🔴 API Unavailable"
            return ("Error: Gemini API not available.", "", "Dependencies missing", 0, mode_status)

        validate_string(user_subject, strip_whitespace=False)
        # Create a more creative and contextual prompt by analyzing the user's intent
        enhanced_subject = self.enhance_user_subject(user_subject, cinematic_style, scene_type)
        if custom_instructions.strip():
            enhanced_subject += f"\n\nAdditional Creative Direction: {custom_instructions.strip()}"

        nsfw_mode = bool(nsfw_mode)
        
        # Create mode status indicator
        mode_status = f"{'🔞 NSFW' if nsfw_mode else '🎬 Normal'} | {cinematic_style} | {wan_model}"
        
        system_prompt = self.get_wan_expert_system_prompt(wan_model, cinematic_style, scene_type, nsfw_mode)
        parts = self.create_gemini_parts(system_prompt, enhanced_subject, reference_image)

        try:
            from comfy_api_nodes.nodes_gemini import get_gemini_endpoint
            endpoint = get_gemini_endpoint(gemini_model)
            response = await SynchronousOperation(
                endpoint=endpoint,
                request=GeminiGenerateContentRequest(contents=[GeminiContent(role="user", parts=parts)]),
                auth_kwargs=kwargs,
            ).execute()

            response_text = ""
            if response.candidates:
                candidate = response.candidates[0]
                if candidate.content and candidate.content.parts:
                    text_parts = [p.text for p in candidate.content.parts if hasattr(p, 'text') and p.text]
                    response_text = "\n".join(text_parts)

            if not response_text:
                return ("Error: Empty response", "–no blurriness", "API returned empty response", 0, mode_status)

            optimized_prompt, negative_prompt, analysis = self.extract_prompt_components(response_text)
            validation = self.validate_prompt_structure(optimized_prompt)
            word_count = validation['word_count']
            analysis += f"\n\nValidation: {word_count} words (target: 80-120)"

            if unique_id:
                PromptServer.instance.send_progress_text(f"Generated {word_count}-word WAN 2.2 prompt", node_id=unique_id)
                
                # Send enhanced UI update message
                PromptServer.instance.send_sync("wan_expert_update", {
                    "node_id": unique_id,
                    "word_count": word_count,
                    "nsfw_mode": nsfw_mode,
                    "style": cinematic_style,
                    "model": wan_model,
                    "validation": validation
                })

            return (optimized_prompt, negative_prompt, analysis, word_count, mode_status)

        except Exception as e:
            return (
                f"API Error - fallback for subject: {user_subject}",
                "–no blurriness, –no artifacts, –no text overlay",
                str(e),
                0,
                f"🔴 Error | {cinematic_style} | {wan_model}"
            )


NODE_CLASS_MAPPINGS = {
    "WAN22ExpertPrompter": WAN22ExpertPrompter,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WAN22ExpertPrompter": "🎬 WAN 2.2 Expert Prompter",
}
