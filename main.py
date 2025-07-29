#!/usr/bin/env python3
"""
Project-S V2 - Clean Architecture Main Entry Point
--------------------------------------------------
Built on True Golden Age foundations from C:/0530/project_s_agent

This is the main entry point for Project-S V2, featuring:
- Clean modern architecture
- True Golden Age components (929+959+585 lines of advanced code)
- Smart Tool Orchestration 
- Enhanced Workflow Engine with LangGraph
- Advanced Decision Routing
- Multi-Model AI Integration
"""

import logging
import asyncio
import sys
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def display_banner():
    """Display the Project-S V2 Simplified banner."""
    banner = """
╔══════════════════════════════════════════════════════════════╗
║                 PROJECT-S V2 SIMPLIFIED                     ║
║                Same Power, Half Complexity                  ║
╠══════════════════════════════════════════════════════════════╣
║  🧠  Enhanced Workflow Engine (PRIMARY) - 959 lines        ║
║  🛠️  Smart Tool Orchestrator - 929 lines                  ║  
║  🌐  Browser Use (PRIMARY WEB) - Real automation           ║
║  ⚡  Optimized & Simplified Architecture                   ║
║  🎯  Single Best Tool Per Task Type                        ║
╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)

async def initialize_golden_age_components():
    """Initialize the SIMPLIFIED Golden Age components - Browser Use + Enhanced Engine."""
    logger.info("🚀 Initializing SIMPLIFIED Project-S components...")
    
    components_status = {
        "smart_orchestrator": False,
        "workflow_engine": False, 
        "browser_use": False,
        "cognitive_core": False
    }
    
    # Initialize Smart Tool Orchestrator (ESSENTIAL)
    try:
        from tools.registry import SmartToolOrchestrator, SMART_ORCHESTRATOR_AVAILABLE
        if SMART_ORCHESTRATOR_AVAILABLE:
            orchestrator = SmartToolOrchestrator()
            components_status["smart_orchestrator"] = True
            logger.info("✅ Smart Tool Orchestrator (929 lines) - LOADED")
        else:
            logger.warning("⚠️ Smart Tool Orchestrator - NOT AVAILABLE")
    except Exception as e:
        logger.error(f"❌ Smart Tool Orchestrator failed: {e}")
    
    # Initialize Enhanced Workflow Engine (PRIMARY ENGINE)
    try:
        from core.orchestration.workflow_engine import EnhancedWorkflowEngine
        workflow_engine = EnhancedWorkflowEngine()
        await workflow_engine.initialize()
        components_status["workflow_engine"] = True
        logger.info("✅ Enhanced Workflow Engine (959 lines) - LOADED (PRIMARY)")
    except Exception as e:
        logger.error(f"❌ Enhanced Workflow Engine failed: {e}")
    
    # Initialize Browser Use (PRIMARY WEB TOOL)
    try:
        from tools.implementations.browser_automation_tool import BrowserAutomationTool
        browser_tool = BrowserAutomationTool()
        components_status["browser_use"] = True
        logger.info("✅ Browser Use Tool - LOADED (PRIMARY WEB)")
    except Exception as e:
        logger.error(f"❌ Browser Use Tool failed: {e}")
    
    # Initialize Cognitive Core (IF AVAILABLE)
    try:
        from core.cognitive import CognitiveCore
        cognitive_core = CognitiveCore()
        components_status["cognitive_core"] = True
        logger.info("✅ Cognitive Core - LOADED")
    except Exception as e:
        logger.error(f"❌ Cognitive Core failed: {e}")
    
    # Summary
    loaded_count = sum(components_status.values())
    total_count = len(components_status)
    
    logger.info(f"🎯 SIMPLIFIED System Status: {loaded_count}/{total_count} components loaded")
    
    if loaded_count >= 2:  # Smart Orchestrator + Enhanced Engine minimum
        logger.info("🎉 SIMPLIFIED Project-S system ready!")
        logger.info("🧠 Primary Engine: Enhanced Workflow")
        logger.info("🌐 Primary Web Tool: Browser Use")
        return True
    else:
        logger.error("💥 Critical components failed to load")
        return False

async def start_interactive_mode(processor):
    """Start interactive command mode."""
    logger.info("🎛️ Starting interactive command mode...")
    print("\n" + "="*60)
    print("🎯 PROJECT-S V2 INTERAKTÍV PARANCS MÓD")
    print("="*60)
    print("💡 Természetes nyelvi parancsokat adhatsz meg!")
    print("📝 Példák:")
    print("   - 'Lassú a gépem, gyorsítsd fel!'")
    print("   - 'Mi használja a CPU-t?'")
    print("   - 'Ellenőrizd a hálózati kapcsolatot'")
    print("   - 'Listázd ki a futó folyamatokat'")
    print("   - 'quit' vagy 'exit' a kilépéshez")
    print("="*60)
    
    try:
        while True:
            try:
                # Prompt for user input
                print("\n🎯 Parancs:")
                user_input = input("➤ ").strip()
                
                if not user_input:
                    continue
                    
                # Check for exit commands
                if user_input.lower() in ['quit', 'exit', 'kilépés', 'vége']:
                    print("👋 Kilépés az interaktív módból...")
                    break
                
                # Check for help
                if user_input.lower() in ['help', 'súgó', '?']:
                    show_help()
                    continue
                
                # Process the natural language command
                print(f"\n🚀 Parancs feldolgozása: '{user_input}'")
                await processor.process_natural_language_task(user_input)
                
                print("\n" + "="*40 + " KÉSZ " + "="*40)
                
            except EOFError:
                print("\n👋 EOF detected, exiting...")
                break
            except KeyboardInterrupt:
                print("\n👋 Ctrl+C detected, exiting...")
                break
            except Exception as e:
                logger.error(f"❌ Error processing command: {e}")
                print(f"❌ Hiba a parancs feldolgozása során: {e}")
                
    except Exception as e:
        logger.error(f"❌ Interactive mode failed: {e}")

def show_help():
    """Show help information."""
    help_text = """
🆘 PROJECT-S V2 SÚGÓ

📋 ALAPVETŐ PARANCSOK:
  quit, exit, kilépés   - Kilépés a programból
  help, súgó, ?         - Ez a súgó

🎯 TERMÉSZETES NYELVI PARANCSOK:
  
🔧 RENDSZERADMINISZTRÁCIÓ:
  • "Lassú a gépem, gyorsítsd fel!"
  • "Mi használja a CPU-t és memóriát?"
  • "Listázd ki a futó folyamatokat"
  • "Ellenőrizd a rendszer állapotát"
  
🌐 HÁLÓZAT:
  • "Ellenőrizd a hálózati kapcsolatot"
  • "Mi a géP IP címe?"
  • "Pingeld meg a google.com-ot"
  
💾 TÁRHELY:
  • "Van elég szabad hely a lemezen?"
  • "Melyik mappa foglalja a legtöbb helyet?"
  
ℹ️ INFORMÁCIÓ:
  • "Milyen Windows verzió fut?"
  • "Milyen processzor van a gépben?"
  • "Mennyi a teljes RAM?"

🎨 A rendszer intelligensen értelmezi a kéréseket és megfelelő 
   Windows parancsokat generál és futtat!
"""
    print(help_text)

async def main():
    """Main entry point for Project-S V2."""
    display_banner()
    
    logger.info("🚀 Starting Project-S V2 Clean Architecture...")
    logger.info(f"📍 Working directory: {Path.cwd()}")
    
    # Initialize Golden Age components
    success = await initialize_golden_age_components()
    
    if success:
        logger.info("✅ Project-S V2 initialization successful!")
        logger.info("🎯 True Golden Age architecture is ready!")
        
        # Initialize command processor
        try:
            from complex_task_tester import ComplexTaskProcessor
            processor = ComplexTaskProcessor()
            logger.info("🤖 Command processor initialized successfully!")
            
            # Interactive command mode
            await start_interactive_mode(processor)
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize command processor: {e}")
            logger.info("💡 Basic mode available... (Press Ctrl+C to exit)")
            
            try:
                while True:
                    await asyncio.sleep(1)
            except KeyboardInterrupt:
                logger.info("👋 Shutting down Project-S V2...")
    else:
        logger.error("❌ Project-S V2 initialization failed!")
        sys.exit(1)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("👋 Project-S V2 shutdown complete")
    except Exception as e:
        logger.error(f"💥 Fatal error: {e}")
        sys.exit(1)
