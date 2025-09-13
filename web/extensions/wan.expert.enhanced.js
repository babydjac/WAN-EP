import { app } from "../../../scripts/app.js";

// Enhanced color schemes with dynamic NSFW switching
const THEMES = {
  NORMAL: {
    primary: "#00E5FF",      // electric cyan
    secondary: "#FF6B35",    // sunset orange
    background: "#0B1F2A",   // deep teal
    accent: "#7C4DFF",       // electric purple
    text: "#E8F4FD",         // ice blue
    glow: "#00E5FF80"        // glowing cyan
  },
  NSFW: {
    primary: "#FF1744",      // hot pink/red
    secondary: "#E91E63",    // deep pink
    background: "#1A0B1E",   // dark purple
    accent: "#FF4081",       // accent pink
    text: "#FFE0F0",         // light pink
    glow: "#FF174480"        // glowing red
  }
};

// Enhanced node styling system
class WanNodeStyler {
  constructor() {
    this.currentTheme = THEMES.NORMAL;
    this.isNsfwMode = false;
  }

  // Inject custom CSS styles
  injectStyles() {
    if (document.getElementById('wan-enhanced-styles')) return;
    
    const style = document.createElement('style');
    style.id = 'wan-enhanced-styles';
    style.textContent = `
      .wan-node-enhanced {
        border-radius: 12px !important;
        box-shadow: 0 0 20px var(--wan-glow) !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        position: relative !important;
        overflow: visible !important;
      }

      .wan-node-enhanced::before {
        content: '';
        position: absolute;
        top: -2px;
        left: -2px;
        right: -2px;
        bottom: -2px;
        background: linear-gradient(45deg, var(--wan-primary), var(--wan-secondary), var(--wan-accent));
        border-radius: 14px;
        z-index: -1;
        opacity: 0.8;
        animation: wan-border-glow 2s ease-in-out infinite alternate;
      }

      @keyframes wan-border-glow {
        0% { opacity: 0.6; transform: scale(1); }
        100% { opacity: 1; transform: scale(1.01); }
      }

      .wan-header {
        background: linear-gradient(135deg, var(--wan-primary)20, var(--wan-secondary)20);
        color: var(--wan-text);
        padding: 8px 12px;
        border-radius: 8px 8px 0 0;
        font-weight: 600;
        font-size: 12px;
        text-shadow: 0 0 8px var(--wan-glow);
        border-bottom: 1px solid var(--wan-primary)40;
      }

      .wan-mode-indicator {
        position: absolute;
        top: 4px;
        right: 4px;
        width: 12px;
        height: 12px;
        border-radius: 50%;
        background: var(--wan-accent);
        box-shadow: 0 0 8px var(--wan-accent);
        animation: wan-pulse 1.5s ease-in-out infinite;
      }

      @keyframes wan-pulse {
        0%, 100% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.7; transform: scale(1.1); }
      }

      .wan-nsfw-indicator {
        background: var(--wan-primary) !important;
        animation: wan-nsfw-pulse 1s ease-in-out infinite !important;
      }

      @keyframes wan-nsfw-pulse {
        0%, 100% { opacity: 1; transform: scale(1); box-shadow: 0 0 12px var(--wan-primary); }
        50% { opacity: 0.8; transform: scale(1.15); box-shadow: 0 0 20px var(--wan-primary); }
      }

      .wan-widget-enhanced {
        background: var(--wan-background)80 !important;
        border: 1px solid var(--wan-primary)40 !important;
        border-radius: 6px !important;
        color: var(--wan-text) !important;
        transition: all 0.2s ease !important;
      }

      .wan-widget-enhanced:hover {
        border-color: var(--wan-primary) !important;
        box-shadow: 0 0 8px var(--wan-glow) !important;
      }

      .wan-widget-enhanced:focus {
        outline: none !important;
        border-color: var(--wan-accent) !important;
        box-shadow: 0 0 12px var(--wan-accent)60 !important;
      }

      .wan-output-socket {
        background: radial-gradient(circle, var(--wan-accent), var(--wan-primary)) !important;
        border: 2px solid var(--wan-text) !important;
        box-shadow: 0 0 8px var(--wan-glow) !important;
        animation: wan-socket-glow 2s ease-in-out infinite alternate !important;
      }

      @keyframes wan-socket-glow {
        0% { box-shadow: 0 0 8px var(--wan-glow); }
        100% { box-shadow: 0 0 16px var(--wan-primary); }
      }

      .wan-floating-panel {
        position: absolute;
        top: -40px;
        left: 50%;
        transform: translateX(-50%);
        background: linear-gradient(135deg, var(--wan-background)95, var(--wan-primary)20);
        border: 1px solid var(--wan-primary)60;
        border-radius: 8px;
        padding: 6px 12px;
        font-size: 11px;
        color: var(--wan-text);
        white-space: nowrap;
        z-index: 1000;
        backdrop-filter: blur(10px);
        box-shadow: 0 4px 20px var(--wan-glow);
        opacity: 0;
        transform: translateX(-50%) translateY(-10px);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
      }

      .wan-floating-panel.visible {
        opacity: 1;
        transform: translateX(-50%) translateY(0);
      }

      .wan-stats-display {
        position: absolute;
        bottom: -30px;
        left: 0;
        right: 0;
        display: flex;
        justify-content: space-around;
        font-size: 10px;
        color: var(--wan-text)80;
      }

      .wan-stat-item {
        background: var(--wan-background)90;
        padding: 2px 6px;
        border-radius: 4px;
        border: 1px solid var(--wan-primary)30;
      }
    `;
    document.head.appendChild(style);
  }

  // Apply theme colors as CSS variables
  applyTheme(node, theme) {
    const root = document.documentElement;
    root.style.setProperty('--wan-primary', theme.primary);
    root.style.setProperty('--wan-secondary', theme.secondary);
    root.style.setProperty('--wan-background', theme.background);
    root.style.setProperty('--wan-accent', theme.accent);
    root.style.setProperty('--wan-text', theme.text);
    root.style.setProperty('--wan-glow', theme.glow);

    // Apply to node
    if (node) {
      node.color = theme.primary;
      node.bgcolor = theme.background;
    }
  }

  // Enhanced node decoration
  enhanceNode(node) {
    if (!node.canvas) return;

    // Add enhanced styling class
    if (node.canvas.canvas) {
      node.canvas.canvas.classList.add('wan-node-enhanced');
    }

    // Create floating info panel
    this.createFloatingPanel(node);
    
    // Create mode indicator
    this.createModeIndicator(node);
    
    // Enhance widgets
    this.enhanceWidgets(node);
    
    // Add stats display
    this.createStatsDisplay(node);

    // Override draw methods for custom rendering
    this.overrideDrawMethods(node);
  }

  createFloatingPanel(node) {
    node.__wanPanel = document.createElement('div');
    node.__wanPanel.className = 'wan-floating-panel';
    node.__wanPanel.innerHTML = '🎬 WAN 2.2 Expert Prompter';
    
    if (node.canvas && node.canvas.canvas) {
      node.canvas.canvas.appendChild(node.__wanPanel);
    }

    // Show/hide on hover
    let hoverTimeout;
    const showPanel = () => {
      clearTimeout(hoverTimeout);
      node.__wanPanel.classList.add('visible');
    };
    const hidePanel = () => {
      hoverTimeout = setTimeout(() => {
        node.__wanPanel.classList.remove('visible');
      }, 300);
    };

    if (node.canvas && node.canvas.canvas) {
      node.canvas.canvas.addEventListener('mouseenter', showPanel);
      node.canvas.canvas.addEventListener('mouseleave', hidePanel);
    }
  }

  createModeIndicator(node) {
    const indicator = document.createElement('div');
    indicator.className = 'wan-mode-indicator';
    node.__modeIndicator = indicator;
    
    if (node.canvas && node.canvas.canvas) {
      node.canvas.canvas.appendChild(indicator);
    }
  }

  createStatsDisplay(node) {
    const stats = document.createElement('div');
    stats.className = 'wan-stats-display';
    stats.innerHTML = `
      <div class="wan-stat-item">Words: <span id="wan-word-count">--</span></div>
      <div class="wan-stat-item">Style: <span id="wan-style">--</span></div>
      <div class="wan-stat-item">Model: <span id="wan-model">--</span></div>
    `;
    node.__statsDisplay = stats;
    
    if (node.canvas && node.canvas.canvas) {
      node.canvas.canvas.appendChild(stats);
    }
  }

  enhanceWidgets(node) {
    if (!node.widgets) return;
    
    node.widgets.forEach(widget => {
      if (widget.element) {
        widget.element.classList.add('wan-widget-enhanced');
      }
    });
  }

  overrideDrawMethods(node) {
    const originalDrawForeground = node.onDrawForeground;
    const originalDrawBackground = node.onDrawBackground;

    // Enhanced foreground drawing
    node.onDrawForeground = function(ctx) {
      if (originalDrawForeground) {
        originalDrawForeground.call(this, ctx);
      }

      // Draw glowing border effect
      ctx.save();
      ctx.strokeStyle = this.__nsfwMode ? THEMES.NSFW.primary : THEMES.NORMAL.primary;
      ctx.lineWidth = 2;
      ctx.shadowColor = this.__nsfwMode ? THEMES.NSFW.glow : THEMES.NORMAL.glow;
      ctx.shadowBlur = 10;
      ctx.beginPath();
      ctx.roundRect(0, 0, this.size[0], this.size[1], 12);
      ctx.stroke();
      ctx.restore();

      // Draw title with glow
      ctx.save();
      ctx.fillStyle = this.__nsfwMode ? THEMES.NSFW.text : THEMES.NORMAL.text;
      ctx.font = "bold 12px Arial";
      ctx.shadowColor = this.__nsfwMode ? THEMES.NSFW.primary : THEMES.NORMAL.primary;
      ctx.shadowBlur = 8;
      ctx.fillText("🎬 WAN 2.2", 10, 20);
      ctx.restore();

      // Draw NSFW indicator if active
      if (this.__nsfwMode) {
        ctx.save();
        ctx.fillStyle = THEMES.NSFW.primary;
        ctx.shadowColor = THEMES.NSFW.primary;
        ctx.shadowBlur = 6;
        ctx.beginPath();
        ctx.arc(this.size[0] - 15, 15, 6, 0, Math.PI * 2);
        ctx.fill();
        ctx.fillStyle = THEMES.NSFW.text;
        ctx.font = "bold 8px Arial";
        ctx.textAlign = "center";
        ctx.fillText("18+", this.size[0] - 15, 18);
        ctx.restore();
      }
    };

    // Enhanced background drawing
    node.onDrawBackground = function(ctx) {
      if (originalDrawBackground) {
        originalDrawBackground.call(this, ctx);
      }

      // Draw animated gradient background
      const gradient = ctx.createLinearGradient(0, 0, this.size[0], this.size[1]);
      const theme = this.__nsfwMode ? THEMES.NSFW : THEMES.NORMAL;
      
      gradient.addColorStop(0, theme.background + "F0");
      gradient.addColorStop(0.5, theme.background + "E0");
      gradient.addColorStop(1, theme.background + "F0");
      
      ctx.fillStyle = gradient;
      ctx.beginPath();
      ctx.roundRect(0, 0, this.size[0], this.size[1], 12);
      ctx.fill();
    };
  }

  // Switch between normal and NSFW modes
  switchMode(node, isNsfw) {
    if (!node) return;
    
    const oldMode = this.isNsfwMode;
    this.isNsfwMode = isNsfw;
    node.__nsfwMode = isNsfw;
    
    // Apply new theme
    const theme = isNsfw ? THEMES.NSFW : THEMES.NORMAL;
    this.applyTheme(node, theme);
    
    // Update mode indicator
    if (node.__modeIndicator) {
      if (isNsfw) {
        node.__modeIndicator.classList.add('wan-nsfw-indicator');
      } else {
        node.__modeIndicator.classList.remove('wan-nsfw-indicator');
      }
    }

    // Update floating panel text
    if (node.__wanPanel) {
      node.__wanPanel.innerHTML = isNsfw ? '🔞 WAN 2.2 NSFW Mode' : '🎬 WAN 2.2 Expert Prompter';
    }

    // Force redraw
    if (node.graph && node.graph.canvas) {
      node.setDirtyCanvas(true, true);
    }

    // Animate transition
    if (oldMode !== isNsfw) {
      this.animateTransition(node, isNsfw);
    }
  }

  animateTransition(node, isNsfw) {
    if (!node.canvas || !node.canvas.canvas) return;
    
    const canvas = node.canvas.canvas;
    canvas.style.transition = 'all 0.5s cubic-bezier(0.4, 0, 0.2, 1)';
    canvas.style.transform = 'scale(1.05)';
    
    setTimeout(() => {
      canvas.style.transform = 'scale(1)';
    }, 200);
    
    setTimeout(() => {
      canvas.style.transition = '';
    }, 500);
  }

  // Update stats display
  updateStats(node, wordCount, style, model) {
    if (!node.__statsDisplay) return;
    
    const wordCountEl = node.__statsDisplay.querySelector('#wan-word-count');
    const styleEl = node.__statsDisplay.querySelector('#wan-style');
    const modelEl = node.__statsDisplay.querySelector('#wan-model');
    
    if (wordCountEl) wordCountEl.textContent = wordCount || '--';
    if (styleEl) styleEl.textContent = style || '--';
    if (modelEl) modelEl.textContent = model || '--';
  }
}

// Initialize the enhanced styler
const wanStyler = new WanNodeStyler();

app.registerExtension({
  name: "wan.expert.prompter.enhanced",
  
  setup() {
    // Inject enhanced styles
    wanStyler.injectStyles();
    
    // Listen for server updates
    app.api.addEventListener("wan_expert_update", (event) => {
      const data = event.detail;
      const node = app.graph.getNodeById(data.node_id);
      if (node && node.comfyClass === "WAN22ExpertPrompter") {
        wanStyler.updateStats(node, data.word_count, data.style, data.model);
        wanStyler.switchMode(node, data.nsfw_mode);
      }
    });
  },

  nodeCreated(node) {
    try {
      if (node?.comfyClass === "WAN22ExpertPrompter") {
        // Apply initial styling
        wanStyler.enhanceNode(node);
        wanStyler.applyTheme(node, THEMES.NORMAL);
        
        // Set up NSFW mode monitoring
        const nsfwWidget = node.widgets?.find(w => w.name === "nsfw_mode");
        if (nsfwWidget) {
          const originalCallback = nsfwWidget.callback;
          nsfwWidget.callback = function(value) {
            wanStyler.switchMode(node, Boolean(value));
            if (originalCallback) originalCallback.call(this, value);
          };
          
          // Initial mode setup
          wanStyler.switchMode(node, Boolean(nsfwWidget.value));
        }

        // Monitor other widgets for stats updates
        const styleWidget = node.widgets?.find(w => w.name === "cinematic_style");
        const modelWidget = node.widgets?.find(w => w.name === "wan_model");
        
        const updateStats = () => {
          const style = styleWidget?.value || '';
          const model = modelWidget?.value || '';
          wanStyler.updateStats(node, '0', style, model);
        };

        if (styleWidget) {
          const origCallback = styleWidget.callback;
          styleWidget.callback = function(value) {
            updateStats();
            if (origCallback) origCallback.call(this, value);
          };
        }

        if (modelWidget) {
          const origCallback = modelWidget.callback;
          modelWidget.callback = function(value) {
            updateStats();
            if (origCallback) origCallback.call(this, value);
          };
        }

        // Monitor execution results for word count updates
        const originalExecuted = node.onExecuted?.bind(node);
        node.onExecuted = function(output) {
          if (output && output.word_count !== undefined) {
            const style = styleWidget?.value || '';
            const model = modelWidget?.value || '';
            wanStyler.updateStats(node, output.word_count, style, model);
          }
          if (originalExecuted) originalExecuted(output);
        };

        // Initial stats update
        updateStats();
      }
    } catch (error) {
      console.error('WAN Enhanced styling error:', error);
    }
  }
});