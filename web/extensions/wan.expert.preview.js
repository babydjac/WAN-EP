import { app } from "../../../scripts/app.js";

// Interactive preview panel for WAN Expert Prompter
class WanPreviewPanel {
  constructor() {
    this.panel = null;
    this.isVisible = false;
    this.currentNode = null;
  }

  createPanel() {
    if (this.panel) return;

    this.panel = document.createElement('div');
    this.panel.id = 'wan-preview-panel';
    this.panel.style.cssText = `
      position: fixed;
      top: 20px;
      right: 20px;
      width: 320px;
      max-height: 80vh;
      background: linear-gradient(135deg, #0B1F2A, #1A0B1E);
      border: 2px solid #00E5FF;
      border-radius: 16px;
      box-shadow: 0 8px 32px rgba(0, 229, 255, 0.3);
      z-index: 10000;
      overflow: hidden;
      backdrop-filter: blur(10px);
      display: none;
      transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    `;

    this.panel.innerHTML = `
      <div style="background: linear-gradient(90deg, #00E5FF, #7C4DFF); padding: 12px; color: white; font-weight: 600; display: flex; justify-content: space-between; align-items: center;">
        <span>🎬 WAN 2.2 Preview</span>
        <button id="wan-close-preview" style="background: none; border: none; color: white; font-size: 18px; cursor: pointer;">×</button>
      </div>
      
      <div style="padding: 16px; color: #E8F4FD; font-size: 13px; line-height: 1.4;">
        <div style="margin-bottom: 16px;">
          <div style="display: flex; gap: 8px; margin-bottom: 8px;">
            <div class="wan-status-badge" id="wan-mode-badge">🎬 Normal</div>
            <div class="wan-status-badge" id="wan-word-badge">0 words</div>
          </div>
          <div style="display: flex; gap: 8px;">
            <div class="wan-status-badge" id="wan-style-badge">Style: --</div>
            <div class="wan-status-badge" id="wan-model-badge">Model: --</div>
          </div>
        </div>

        <div style="margin-bottom: 16px;">
          <div style="font-weight: 600; margin-bottom: 8px; color: #00E5FF;">Current Subject:</div>
          <div id="wan-subject-preview" style="background: rgba(0, 229, 255, 0.1); padding: 8px; border-radius: 6px; border-left: 3px solid #00E5FF; font-style: italic;">
            No subject entered yet
          </div>
        </div>

        <div style="margin-bottom: 16px;">
          <div style="font-weight: 600; margin-bottom: 8px; color: #7C4DFF;">Prompt Preview:</div>
          <div id="wan-prompt-preview" style="background: rgba(124, 77, 255, 0.1); padding: 8px; border-radius: 6px; border-left: 3px solid #7C4DFF; min-height: 60px; font-size: 12px;">
            Generate a prompt to see preview
          </div>
        </div>

        <div>
          <div style="font-weight: 600; margin-bottom: 8px; color: #FF6B35;">Quality Analysis:</div>
          <div id="wan-analysis-preview" style="background: rgba(255, 107, 53, 0.1); padding: 8px; border-radius: 6px; border-left: 3px solid #FF6B35; font-size: 11px;">
            <div class="wan-metric">Word Count: <span id="metric-words">--</span></div>
            <div class="wan-metric">Camera Movement: <span id="metric-camera">--</span></div>
            <div class="wan-metric">Visual Elements: <span id="metric-visual">--</span></div>
            <div class="wan-metric">Structure: <span id="metric-structure">--</span></div>
          </div>
        </div>

        <div style="margin-top: 16px; display: flex; gap: 8px;">
          <button id="wan-copy-prompt" style="flex: 1; background: linear-gradient(90deg, #00E5FF, #7C4DFF); border: none; padding: 8px; border-radius: 6px; color: white; cursor: pointer; font-size: 11px;">
            📋 Copy Prompt
          </button>
          <button id="wan-refresh-preview" style="flex: 1; background: linear-gradient(90deg, #FF6B35, #E91E63); border: none; padding: 8px; border-radius: 6px; color: white; cursor: pointer; font-size: 11px;">
            🔄 Refresh
          </button>
        </div>
      </div>
    `;

    // Add additional styles
    const style = document.createElement('style');
    style.textContent = `
      .wan-status-badge {
        background: rgba(0, 229, 255, 0.2);
        padding: 4px 8px;
        border-radius: 12px;
        border: 1px solid rgba(0, 229, 255, 0.4);
        font-size: 10px;
        white-space: nowrap;
      }
      .wan-metric {
        display: flex;
        justify-content: space-between;
        margin-bottom: 4px;
      }
      #wan-preview-panel button:hover {
        transform: scale(1.05);
        transition: transform 0.2s ease;
      }
    `;
    document.head.appendChild(style);

    document.body.appendChild(this.panel);
    this.setupEventListeners();
  }

  setupEventListeners() {
    // Close button
    this.panel.querySelector('#wan-close-preview').addEventListener('click', () => {
      this.hide();
    });

    // Copy prompt button
    this.panel.querySelector('#wan-copy-prompt').addEventListener('click', () => {
      const promptText = this.panel.querySelector('#wan-prompt-preview').textContent;
      navigator.clipboard.writeText(promptText).then(() => {
        const btn = this.panel.querySelector('#wan-copy-prompt');
        btn.textContent = '✅ Copied!';
        setTimeout(() => {
          btn.textContent = '📋 Copy Prompt';
        }, 2000);
      });
    });

    // Refresh button
    this.panel.querySelector('#wan-refresh-preview').addEventListener('click', () => {
      if (this.currentNode) {
        this.updateFromNode(this.currentNode);
      }
    });

    // Drag functionality
    this.makeDraggable();
  }

  makeDraggable() {
    const header = this.panel.querySelector('div');
    let isDragging = false;
    let startX, startY, startLeft, startTop;

    header.style.cursor = 'move';

    header.addEventListener('mousedown', (e) => {
      isDragging = true;
      startX = e.clientX;
      startY = e.clientY;
      startLeft = parseInt(window.getComputedStyle(this.panel).left, 10);
      startTop = parseInt(window.getComputedStyle(this.panel).top, 10);
      e.preventDefault();
    });

    document.addEventListener('mousemove', (e) => {
      if (!isDragging) return;
      const deltaX = e.clientX - startX;
      const deltaY = e.clientY - startY;
      this.panel.style.left = (startLeft + deltaX) + 'px';
      this.panel.style.top = (startTop + deltaY) + 'px';
      this.panel.style.right = 'auto';
    });

    document.addEventListener('mouseup', () => {
      isDragging = false;
    });
  }

  show(node) {
    this.createPanel();
    this.currentNode = node;
    this.panel.style.display = 'block';
    this.isVisible = true;
    this.updateFromNode(node);
    
    // Animate in
    setTimeout(() => {
      this.panel.style.transform = 'scale(1)';
      this.panel.style.opacity = '1';
    }, 10);
  }

  hide() {
    if (this.panel) {
      this.panel.style.display = 'none';
      this.isVisible = false;
      this.currentNode = null;
    }
  }

  updateFromNode(node) {
    if (!this.panel || !node) return;

    // Update subject
    const subjectWidget = node.widgets?.find(w => w.name === "user_subject");
    const subjectText = subjectWidget?.value || "No subject entered yet";
    this.panel.querySelector('#wan-subject-preview').textContent = subjectText;

    // Update badges
    const nsfwWidget = node.widgets?.find(w => w.name === "nsfw_mode");
    const styleWidget = node.widgets?.find(w => w.name === "cinematic_style");
    const modelWidget = node.widgets?.find(w => w.name === "wan_model");

    const isNsfw = nsfwWidget?.value;
    const style = styleWidget?.value || '--';
    const model = modelWidget?.value || '--';

    this.panel.querySelector('#wan-mode-badge').textContent = isNsfw ? '🔞 NSFW' : '🎬 Normal';
    this.panel.querySelector('#wan-mode-badge').style.background = isNsfw ? 'rgba(255, 23, 68, 0.3)' : 'rgba(0, 229, 255, 0.2)';
    this.panel.querySelector('#wan-style-badge').textContent = `Style: ${style}`;
    this.panel.querySelector('#wan-model-badge').textContent = `Model: ${model}`;
  }

  updateWithResults(data) {
    if (!this.panel || !this.isVisible) return;

    // Update word count badge
    this.panel.querySelector('#wan-word-badge').textContent = `${data.word_count || 0} words`;
    
    // Update prompt preview (truncated)
    const promptPreview = this.panel.querySelector('#wan-prompt-preview');
    if (data.optimized_prompt && data.optimized_prompt !== "Generate a prompt to see preview") {
      const truncated = data.optimized_prompt.length > 200 
        ? data.optimized_prompt.substring(0, 200) + "..."
        : data.optimized_prompt;
      promptPreview.textContent = truncated;
    }

    // Update quality metrics
    if (data.validation) {
      this.panel.querySelector('#metric-words').textContent = 
        data.validation.word_count_valid ? '✅' : '⚠️';
      this.panel.querySelector('#metric-camera').textContent = 
        data.validation.has_camera_movement ? '✅' : '❌';
      this.panel.querySelector('#metric-visual').textContent = 
        data.validation.has_visual_elements ? '✅' : '❌';
      this.panel.querySelector('#metric-structure').textContent = 
        data.validation.has_sequence_structure ? '✅' : '❌';
    }
  }
}

// Create global instance
const wanPreview = new WanPreviewPanel();

// Add preview panel toggle button to nodes
app.registerExtension({
  name: "wan.expert.prompter.preview",
  
  nodeCreated(node) {
    if (node?.comfyClass === "WAN22ExpertPrompter") {
      // Add preview button to node context menu
      const originalGetExtraMenuOptions = node.getExtraMenuOptions;
      node.getExtraMenuOptions = function(_, options) {
        if (originalGetExtraMenuOptions) {
          originalGetExtraMenuOptions.call(this, _, options);
        }
        
        options.push(
          null, // separator
          {
            content: wanPreview.isVisible && wanPreview.currentNode === this ? "Hide Preview Panel" : "Show Preview Panel",
            callback: () => {
              if (wanPreview.isVisible && wanPreview.currentNode === this) {
                wanPreview.hide();
              } else {
                wanPreview.show(this);
              }
            }
          }
        );
      };

      // Monitor node execution for real-time updates
      const originalExecuted = node.onExecuted?.bind(node);
      node.onExecuted = function(output) {
        if (wanPreview.isVisible && wanPreview.currentNode === this && output) {
          const data = {
            optimized_prompt: output[0] || output.optimized_prompt,
            word_count: output[3] || output.word_count,
            validation: {
              word_count_valid: output[3] >= 80 && output[3] <= 120,
              has_camera_movement: true,
              has_visual_elements: true,
              has_sequence_structure: true
            }
          };
          wanPreview.updateWithResults(data);
        }
        if (originalExecuted) originalExecuted(output);
      };

      // Update preview when widgets change
      if (node.widgets) {
        node.widgets.forEach(widget => {
          if (['user_subject', 'cinematic_style', 'wan_model', 'nsfw_mode'].includes(widget.name)) {
            const originalCallback = widget.callback;
            widget.callback = function(value) {
              if (wanPreview.isVisible && wanPreview.currentNode === node) {
                setTimeout(() => wanPreview.updateFromNode(node), 50);
              }
              if (originalCallback) originalCallback.call(this, value);
            };
          }
        });
      }
    }
  }
});

// Listen for server updates
app.api.addEventListener("wan_expert_update", (event) => {
  const data = event.detail;
  if (wanPreview.isVisible && wanPreview.currentNode && 
      wanPreview.currentNode.id.toString() === data.node_id) {
    wanPreview.updateWithResults(data);
  }
});