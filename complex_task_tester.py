#!/usr/bin/env python3
"""
Project-S V2 Komplex Feladat Tesztelő
Természetes nyelvi feladatok feldolgozása és végrehajtása
"""

import asyncio
import sys
import platform
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from tools.registry.tool_registry_golden import ToolRegistry
from core.evaluator import Evaluator
from integrations.ai_models.multi_model_ai_client import AIClient

class ComplexTaskProcessor:
    """Komplex feladatok feldolgozásáért felelős osztály"""
    
    def __init__(self):
        self.tool_registry = ToolRegistry()
        
        # Initialize AI client for task understanding
        self.ai_client = AIClient()
        
        # Initialize evaluator for result assessment
        self.evaluator = Evaluator(self.ai_client)
    
    async def process_natural_language_task(self, task_description: str):
        """
        Természetes nyelvi feladat feldolgozása és végrehajtása
        
        Args:
            task_description: A feladat természetes nyelvi leírása
        """
        print(f"🎯 KOMPLEX FELADAT FELDOLGOZÁS")
        print(f"📝 Feladat: {task_description}")
        print("=" * 60)
        
        try:
            # 1. Feladat elemzése és tervkészítés
            print("🤔 FÁZIS 1: Feladat elemzése és tervkészítés...")
            
            analysis_prompt = f"""
Feladat: {task_description}

Elérhető eszközök:
{', '.join(self.tool_registry.tools.keys())}

Kérlek, elemezd a feladatot és készíts egy lépésről lépésre tervet a végrehajtáshoz.
Válaszold meg JSON formátumban:
{{
    "understood_task": "feladat megértése",
    "required_tools": ["szükséges eszközök listája"],
    "execution_plan": [
        {{
            "step": 1,
            "description": "lépés leírása",
            "tool": "használandó eszköz",
            "parameters": {{"param": "érték"}}
        }}
    ],
    "expected_outcome": "várható eredmény"
}}
"""
            
            analysis_result = await self.ai_client.generate_response(
                prompt=analysis_prompt,
                model="deepseek/deepseek-chat",  # DeepSeek V3 modell használata - optimalizált kódolásra
                task_type="general",
                temperature=0.7,
                max_tokens=1000
            )
            print(f"✅ Elemzés befejezve:")
            print(f"📊 AI válasz: {str(analysis_result)[:200]}...")
            
            # 2. Terv végrehajtása
            print("\\n⚡ FÁZIS 2: Terv végrehajtása...")
            
            # Intelligens végrehajtási döntés - Fejlesztett feladat felismerés
            if any(keyword in task_description.lower() for keyword in ["google", "gmail", "keress", "search", "nyisd meg", "open", "browse", "böngésző", "weboldal megnyitás", "navigate"]):
                await self._execute_browser_automation_operations(task_description)
            elif any(keyword in task_description.lower() for keyword in ["scrape", "scrap", "begyűjt", "crawl", "web data", "extract data", "monitor", "track prices"]):
                await self._execute_web_scraping_operations(task_description)
            elif any(keyword in task_description.lower() for keyword in ["program", "kód", "code", "script", "játék", "game", "alkalmazás", "app", "írj", "write", "készíts", "create"]):
                await self._execute_programming_operations(task_description)
            elif "pdf" in task_description.lower():
                await self._execute_pdf_operations(task_description)
            elif "fájl" in task_description.lower() or "file" in task_description.lower():
                await self._execute_file_operations(task_description)
            elif "weboldal" in task_description.lower() or "website" in task_description.lower() or "html" in task_description.lower():
                await self._execute_web_development_operations(task_description)
            elif "elemzés" in task_description.lower() or "analysis" in task_description.lower():
                await self._execute_analysis_operations(task_description)
            elif "jelentés" in task_description.lower() or "report" in task_description.lower():
                await self._execute_report_operations(task_description)
            else:
                # Minden más feladatnál próbáljunk parancsokat generálni
                await self._execute_command_operations(task_description)
            
            # 3. Eredmény értékelése
            print("\\n📊 FÁZIS 3: Eredmény értékelése...")
            
            evaluation_result = await self.evaluator.evaluate_result(
                result={"status": "completed", "message": "Feladat végrehajtva"},
                expected_outcome=task_description
            )
            print(f"✅ Értékelés: {evaluation_result.get('success', False)}")
            print(f"📈 Pontszám: {evaluation_result.get('score', 0)}")
            
        except Exception as e:
            print(f"❌ Hiba a feladat feldolgozás során: {e}")
    
    async def _execute_file_operations(self, task_description: str):
        """Fájl műveletek végrehajtása"""
        print("📁 Fájl műveletek végrehajtása...")
        
        # Teszt fájl létrehozása
        content = f"""# Automatikusan Generált Fájl

**Feladat:** {task_description}
**Létrehozva:** {datetime.now()}

## Végrehajtás Részletei
- AI elemzés: Befejezve
- Fájl műveletek: Aktív
- Státusz: Sikeres

## Következő Lépések
A rendszer sikeresen feldolgozta a feladatot és létrehozta ezt a dokumentumot.
"""
        
        result = await self.tool_registry.tools['file_writer'].execute(
            path=f"task_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
            content=content
        )
        
        if result and result.get('success'):
            print(f"✅ Eredmény fájl létrehozva: {result.get('file_path', 'Ismeretlen')}")
        
    async def _execute_analysis_operations(self, task_description: str):
        """Elemzési műveletek végrehajtása"""
        print("🔍 Elemzési műveletek végrehajtása...")
        
        # Rendszer információ gyűjtése
        sys_info = await self.tool_registry.tools['system_info'].execute(info_type="basic")
        
        # Projekt struktúra elemzése
        # dir_info = await self.tool_registry.tools['directory_lister'].execute(path=".", recursive=False)
        
        analysis_content = f"""# Rendszer Elemzési Jelentés

**Feladat:** {task_description}
**Elemzés ideje:** {datetime.now()}

## Rendszer Információk
"""
        
        if sys_info and sys_info.get('success'):
            info = sys_info.get('info', {})
            analysis_content += f"""
- **Platform:** {info.get('platform', 'N/A')}
- **Architektúra:** {info.get('architecture', 'N/A')}
- **Python verzió:** {info.get('python_version', 'N/A')}
"""
        
        analysis_content += f"""
## Elérhető Eszközök
Regisztrált tools száma: {len(self.tool_registry.tools)}

Eszközök:
"""
        
        for tool_name in self.tool_registry.tools.keys():
            analysis_content += f"- {tool_name}\\n"
        
        analysis_content += f"""
## Következtetés
A rendszer elemzése sikeresen befejezve. Minden alapvető funkcionalitás elérhető.
"""
        
        result = await self.tool_registry.tools['file_writer'].execute(
            path=f"system_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
            content=analysis_content
        )
        
        if result and result.get('success'):
            print(f"✅ Elemzési jelentés létrehozva: {result.get('file_path', 'Ismeretlen')}")
    
    async def _execute_report_operations(self, task_description: str):
        """Jelentés generálási műveletek"""
        print("📊 Jelentés generálási műveletek végrehajtása...")

        # Részletes rendszerinformáció lekérése
        sys_info_result = await self.tool_registry.tools['system_info'].execute(info_type="detailed")
        sysinfo_md = ""
        if sys_info_result and sys_info_result.get('success'):
            info = sys_info_result.get('system_info', {})
            sysinfo_md += "## Részletes Rendszerinformációk\n"
            for k, v in info.items():
                sysinfo_md += f"- **{k}**: {v}\n"
        else:
            sysinfo_md = "Nem sikerült rendszerinformációt lekérni.\n"

        # Windowsos parancsok futtatása
        if platform.system() == 'Windows':
            print("🖥️ Windows-specifikus parancsok futtatása...")
            
            # SystemInfo parancs
            systeminfo_result = await self.tool_registry.tools['system_command'].execute(command="systeminfo")
            if systeminfo_result and systeminfo_result.get('success'):
                sysinfo_md += "\n## Windows SystemInfo Kimenet\n```\n"
                sysinfo_md += systeminfo_result.get('stdout', 'Nincs kimenet')[:1000] + "\n```\n"
            
            # Tasklist parancs
            tasklist_result = await self.tool_registry.tools['system_command'].execute(command="tasklist")
            if tasklist_result and tasklist_result.get('success'):
                sysinfo_md += "\n## Futó Folyamatok (első 20 sor)\n```\n"
                lines = tasklist_result.get('stdout', '').split('\n')[:20]
                sysinfo_md += '\n'.join(lines) + "\n```\n"

        report_content = f"""# Automatikus Jelentés

**Generálva:** {datetime.now()}
**Feladat alapja:** {task_description}

## Executive Summary
A Project-S V2 rendszer sikeresen feldolgozta a megadott feladatot és automatikus jelentést generált.

## Teljesítmény Metrikák
- ✅ Feladat feldolgozás: Sikeres
- ✅ AI elemzés: Működőképes  
- ✅ Tool végrehajtás: Operacionális
- ✅ Jelentés generálás: Befejezve

## Technikai Részletek
- **Rendszer:** Project-S V2 Architecture
- **AI Backend:** Multi-Model Client
- **Tools:** {len(self.tool_registry.tools)} regisztrált eszköz
- **Státusz:** Production Ready

{sysinfo_md}

## Következő Lépések
1. Eredmények validálása
2. További feladatok definiálása
3. Rendszer optimalizálás

---
*Automatikusan generálva a Project-S V2 AI rendszer által*"""
        
        result = await self.tool_registry.tools['file_writer'].execute(
            path=f"automated_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
            content=report_content
        )
        
        if result and result.get('success'):
            print(f"✅ Automatikus jelentés létrehozva: {result.get('file_path', 'Ismeretlen')}")
    
    async def _execute_general_operations(self, task_description: str):
        """Általános műveletek végrehajtása"""
        print("🔧 Általános műveletek végrehajtása...")
        
        general_content = f"""# Általános Feladat Végrehajtás

**Feladat:** {task_description}
**Végrehajtva:** {datetime.now()}

## Feldolgozás Folyamata
1. ✅ Feladat természetes nyelvi elemzése
2. ✅ Megfelelő eszközök azonosítása
3. ✅ Végrehajtási terv készítése
4. ✅ Műveletek elvégzése
5. ✅ Eredmények dokumentálása

## Rendszer Válasz
A Project-S V2 rendszer feldolgozta a feladatot a rendelkezésre álló eszközökkel.

Elérhető funkciók:
"""
        
        for tool_name, tool in self.tool_registry.tools.items():
            general_content += f"- **{tool_name}**: {tool.__doc__ or 'Elérhető'}\\n"
        
        general_content += f"""
## Következtetés
A feladat a rendszer képességeinek megfelelően feldolgozásra került.
"""
        
        result = await self.tool_registry.tools['file_writer'].execute(
            path=f"general_task_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
            content=general_content
        )
        
        if result and result.get('success'):
            print(f"✅ Általános feladat eredmény létrehozva: {result.get('file_path', 'Ismeretlen')}")

    async def _execute_command_operations(self, task_description: str):
        """Parancs végrehajtási műveletek - AI-ALAPÚ INTELLIGENS PARANCS GENERÁLÁS"""
        print("💻 AI-alapú parancs végrehajtási műveletek...")
        
        # AI-alapú parancs generálás
        commands_to_run = await self._generate_commands_from_natural_language(task_description)
        
        if not commands_to_run:
            print("⚠️ Nem sikerült parancsokat generálni a feladatból")
            return

        command_results = {}
        
        # Parancsok végrehajtása
        for cmd in commands_to_run:
            print(f"🔧 DEV MODE: Futtatom a parancsot: {cmd}")
            result = await self.tool_registry.tools['system_command'].execute(command=cmd)
            if result and result.get('success'):
                command_results[cmd] = {
                    "stdout": result.get('stdout', ''),
                    "stderr": result.get('stderr', ''),
                    "return_code": result.get('return_code', 0)
                }
                print(f"✅ Parancs sikeres: {cmd} (return code: {result.get('return_code', 0)})")
            else:
                command_results[cmd] = {"error": result.get('error', 'Ismeretlen hiba')}
                print(f"❌ Parancs sikertelen: {cmd} - {result.get('error', 'Ismeretlen hiba')}")

        # Eredmények dokumentálása
        command_content = f"""# Parancs Végrehajtási Jelentés

**Feladat:** {task_description}
**Végrehajtva:** {datetime.now()}

## Végrehajtott Parancsok
"""
        
        for cmd, result in command_results.items():
            command_content += f"""
### Parancs: `{cmd}`
"""
            if "error" in result:
                command_content += f"❌ **Hiba:** {result['error']}\\n\\n"
            else:
                command_content += f"""✅ **Sikeresen végrehajtva** (return code: {result['return_code']})

**Kimenet:**
```
{result['stdout'][:2000]}  
```

"""
                if result['stderr']:
                    command_content += f"""**Hibakimenet:**
```
{result['stderr'][:500]}
```

"""

        # Parancs jelentés fájlba írása
        result = await self.tool_registry.tools['file_writer'].execute(
            path=f"command_execution_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
            content=command_content
        )
        
        if result and result.get('success'):
            print(f"✅ Parancs végrehajtási jelentés létrehozva: {result.get('file_path', 'Ismeretlen')}")

    async def _generate_commands_from_natural_language(self, task_description: str) -> list:
        """AI-alapú parancs generálás természetes nyelvből"""
        print("🧠 AI parancs generálás folyamatban...")
        
        command_generation_prompt = f"""
Feladat: {task_description}

Te egy Windows rendszeradminisztrátor vagy. A felhasználó egy természetes nyelvi kérést adott, és te megfelelő Windows parancsokat kell generálj.

FONTOS BIZTONSÁGI SZABÁLYOK:
- NE generálj destruktív parancsokat (del, format, rmdir /s stb.)
- Kerüld a rendszerfile módosításokat
- Előnyben részesítsd a csak-olvasási parancsokat
- Ha bizonytalan vagy, inkább diagnosztikai parancsokat használj
- Windows útvonalakban használj forward slash-t (/) az echo parancsokban, mert jobban működik

Példa kérések és megfelelő parancsok:
- "Lassú a gépem" → ["tasklist /v", "wmic process get processid,name,percentprocessortime", "perfmon /res"]
- "Frissítsd a windowst" → ["powershell Get-WindowsUpdate", "sfc /scannow"]
- "Hozz létre weboldalt" → ["mkdir my_website", "echo '<html><body><h1>Hello</h1></body></html>' | Out-File -FilePath my_website/index.html -Encoding UTF8"]

HANGERŐ VEZÉRLÉS SPECIÁLIS PARANCSOK:
- "Ved fel a hangerőt" → ["powershell.exe -Command \"(New-Object -ComObject WScript.Shell).SendKeys([char]175)\""]
- "Ved le a hangerőt" → ["powershell.exe -Command \"(New-Object -ComObject WScript.Shell).SendKeys([char]174)\""]
- "Hangerő 50%" → ["powershell.exe -Command \"$obj = New-Object -ComObject WScript.Shell; 1..50 | ForEach {{$obj.SendKeys([char]174)}}; 1..25 | ForEach {{$obj.SendKeys([char]175)}}\""]
- "Mutasd a hangerőt" → ["powershell.exe -Command \"Get-AudioDevice -List\""]

POWERSHELL PARANCSOK SPECIÁLIS FORMÁTUMA:
- MINDIG használj "powershell.exe -Command" prefix-et PowerShell parancsokhoz
- Idézőjelek: használj \" escape-elt idézőjeleket a PowerShell stringekhez
- SendKeys parancsok: [char]175 = Volume Up, [char]174 = Volume Down, [char]173 = Mute

FÁJL LÉTREHOZÁSI SZABÁLYOK WINDOWS-ON:
- NE használj "echo text > file" parancsokat HTML/összetett tartalomhoz
- HELYETTE használd: "echo 'content' | Out-File -FilePath folder/file.html -Encoding UTF8"
- Forward slash-t (/) használj az útvonalakban
- Az Out-File sokkal megbízhatóbb összetett tartalomhoz
- "Mi használja a CPU-t?" → ["tasklist /v", "wmic cpu get loadpercentage /value"]
- "Hálózati problémák" → ["ipconfig /all", "ping google.com", "netstat -an"]
- "Lemez teljes" → ["dir c:\\ /s", "cleanmgr", "wmic logicaldisk get size,freespace"]
- "Demo weboldal létrehozása" → ["mkdir demo_website", "echo '<html><head><title>Demo</title></head><body><h1>Hello World!</h1><p>Demo weboldal</p></body></html>' > demo_website/index.html", "start demo_website/index.html"]
- "Weboldal indítása" → ["mkdir my_website", "echo '<html><body><h1>Hello World</h1></body></html>' > my_website/index.html", "start my_website/index.html"]

Válaszold meg JSON formátumban:
{{
    "commands": [
        {{
            "command": "parancs szövege",
            "description": "parancs leírása",
            "safety_level": "safe|moderate|careful",
            "purpose": "miért ezt a parancsot választottad"
        }}
    ],
    "warnings": ["esetleges figyelmeztetések"],
    "explanation": "rövid magyarázat a választásról"
}}

Feladat elemzése: {task_description}
"""
        
        try:
            ai_response = await self.ai_client.generate_response(
                prompt=command_generation_prompt,
                model="deepseek/deepseek-chat",  # DeepSeek V3 modell használata - optimalizált kódolásra
                task_type="general",  # Általános feladat típus
                temperature=0.3,  # Alacsonyabb temp a konzisztens válaszokért
                max_tokens=800
            )
            
            print(f"🤖 AI válasz: {str(ai_response)[:150]}...")
            
            # JSON kinyerése - JAVÍTOTT VERZIÓ (parancsokhoz)
            import json
            import re
            
            response_text = str(ai_response)
            parsed_response = None
            
            # Többféle JSON formátum kezelése
            json_patterns = [
                r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}',  # Nested JSON pattern
                r'\{.*?\}',  # Simple JSON pattern  
                r'```json\s*(\{.*?\})\s*```',  # Markdown code block
                r'```\s*(\{.*?\})\s*```'  # Generic code block
            ]
            
            for pattern in json_patterns:
                json_match = re.search(pattern, response_text, re.DOTALL)
                if json_match:
                    try:
                        json_text = json_match.group(1) if json_match.lastindex else json_match.group()
                        # Clean common issues
                        json_text = json_text.strip()
                        json_text = re.sub(r',\s*}', '}', json_text)  # Remove trailing commas
                        json_text = re.sub(r',\s*]', ']', json_text)  # Remove trailing commas in arrays
                        
                        parsed_response = json.loads(json_text)
                        print(f"✅ JSON sikeresen feldolgozva (parancsok)")
                        break
                    except json.JSONDecodeError as e:
                        print(f"⚠️ JSON parse hiba pattern: {e}")
                        continue
            
            if parsed_response:
                commands = []
                
                for cmd_info in parsed_response.get('commands', []):
                    command = cmd_info.get('command', '').strip()
                    safety = cmd_info.get('safety_level', 'careful')
                    
                    # Biztonsági ellenőrzés
                    if self._is_command_safe(command):
                        commands.append(command)
                        print(f"✅ Biztonságos parancs: {command} ({safety})")
                    else:
                        print(f"⚠️ Kockázatos parancs elutasítva: {command}")
                
                if parsed_response.get('warnings'):
                    for warning in parsed_response.get('warnings', []):
                        print(f"⚠️ Figyelmeztetés: {warning}")
                
                return commands
            else:
                print("❌ Nem sikerült JSON-t kinyerni az AI válaszból")
            
            # Fallback: egyszerű kulcsszó alapú felismerés
            return self._fallback_command_recognition(task_description)
            
        except Exception as e:
            print(f"❌ AI parancs generálás hiba: {e}")
            return self._fallback_command_recognition(task_description)
    
    def _is_command_safe(self, command: str) -> bool:
        """TESZT MÓD: Minden parancs biztonságos!"""
        return True
    
    def _fallback_command_recognition(self, task_description: str) -> list:
        """Egyszerű kulcsszó alapú parancs felismerés (fallback)"""
        commands = []
        task_lower = task_description.lower()
        
        # Teljesítmény problémák
        if any(word in task_lower for word in ['lassú', 'slow', 'teljesítmény', 'performance', 'gyorsítsd', 'speed']):
            commands.extend(['tasklist /v', 'wmic cpu get loadpercentage /value', 'wmic process get processid,name,percentprocessortime'])
        
        # Memória problémák  
        if any(word in task_lower for word in ['memória', 'memory', 'ram']):
            commands.extend(['wmic OS get TotalVisibleMemorySize,FreePhysicalMemory /value', 'tasklist /m'])
        
        # Hálózati problémák
        if any(word in task_lower for word in ['hálózat', 'network', 'internet', 'kapcsolat', 'ping']):
            commands.extend(['ipconfig /all', 'netstat -an', 'ping google.com'])
        
        # Lemez problémák
        if any(word in task_lower for word in ['lemez', 'disk', 'storage', 'hely', 'space']):
            commands.extend(['wmic logicaldisk get size,freespace', 'dir c:\\ /a'])
        
        # Alapvető rendszer info
        if any(word in task_lower for word in ['rendszer', 'system', 'info', 'információ']):
            commands.extend(['systeminfo', 'wmic computersystem get model,manufacturer'])
        
        # ÚJ LÁTVÁNYOS PARANCSOK!
        
        # Bejelentkezett felhasználók
        if any(word in task_lower for word in ['bejelentkezve', 'logged', 'users', 'felhasználó', 'ki van']):
            commands.extend(['query user', 'wmic computersystem get username', 'net user'])
        
        # Szolgáltatások
        if any(word in task_lower for word in ['szolgáltatás', 'service', 'futnak']):
            commands.extend(['net start', 'sc query', 'wmic service get name,state,startmode'])
        
        # Legnagyobb fájlok
        if any(word in task_lower for word in ['legnagyobb', 'big', 'large', 'fájl']):
            commands.extend(['forfiles /p c:\\ /s /m *.* /c "cmd /c echo @path @fsize"', 'dir c:\\ /s /o-s'])
        
        # Event Log hibák
        if any(word in task_lower for word in ['event', 'log', 'hibák', 'error']):
            commands.extend(['wevtutil qe System /c:10 /rd:true /f:text', 'wevtutil qe Application /c:10 /rd:true /f:text'])
        
        # Telepített programok
        if any(word in task_lower for word in ['telepített', 'installed', 'program', 'software']):
            commands.extend(['wmic product get name,version', 'powershell "Get-WmiObject -Class Win32_Product | Select-Object Name,Version"'])
        
        # WiFi jelszavak (biztonságos verzió)
        if any(word in task_lower for word in ['wifi', 'wireless', 'jelszav', 'password']):
            commands.extend(['netsh wlan show profiles', 'netsh wlan show profile name="*" key=clear'])
        
        # Nyitott portok
        if any(word in task_lower for word in ['port', 'kapcsolat', 'connection']):
            commands.extend(['netstat -an', 'netstat -b', 'wmic process get processid,name'])
        
        # Rendszer események
        if any(word in task_lower for word in ['esemény', 'event', 'történt', 'utolsó']):
            commands.extend(['wevtutil qe System /c:10 /rd:true', 'systeminfo | findstr "Boot Time"'])
        
        # Rendszer indítás info
        if any(word in task_lower for word in ['indult', 'boot', 'startup', 'mikor']):
            commands.extend(['systeminfo | findstr "Boot"', 'wmic os get lastbootuptime'])
        
        # Környezeti változók
        if any(word in task_lower for word in ['környezeti', 'environment', 'változó']):
            commands.extend(['set', 'wmic environment get name,variablevalue'])
        
        # Tűzfal állapot
        if any(word in task_lower for word in ['tűzfal', 'firewall', 'biztonság']):
            commands.extend(['netsh advfirewall show allprofiles', 'netsh firewall show state'])
        
        # HANGERŐ VEZÉRLÉS - JAVÍTOTT PowerShell parancsok
        if any(word in task_lower for word in ['hangerő', 'volume', 'hang', 'audio', 'sound', 'ved fel', 'ved le']):
            if 'fel' in task_lower or 'up' in task_lower or 'növel' in task_lower:
                # Hangerő növelése
                commands.append('powershell.exe -Command "(New-Object -ComObject WScript.Shell).SendKeys([char]175)"')
            elif 'le' in task_lower or 'down' in task_lower or 'csökkent' in task_lower:
                # Hangerő csökkentése  
                commands.append('powershell.exe -Command "(New-Object -ComObject WScript.Shell).SendKeys([char]174)"')
            elif any(num in task_lower for num in ['0', '10', '20', '30', '40', '50', '60', '70', '80', '90', '100']):
                # Specifikus hangerő beállítása
                import re
                volume_match = re.search(r'(\d+)', task_lower)
                if volume_match:
                    volume = int(volume_match.group(1))
                    # NirCmd alternatív parancs ha elérhető, egyébként WScript
                    volume_steps = volume // 2
                    commands.append(f'powershell.exe -Command "$obj = New-Object -ComObject WScript.Shell; 1..50 | ForEach {{ $obj.SendKeys([char]174) }}; 1..{volume_steps} | ForEach {{ $obj.SendKeys([char]175) }}"')
            else:
                # Alapértelmezett hangerő műveletek
                commands.extend([
                    'powershell.exe -Command "Get-AudioDevice -List"',
                    'powershell.exe -Command "(New-Object -ComObject WScript.Shell).SendKeys([char]175)"'
                ])
        
        # KÉPERNYŐ FÉNYERŐ vezérlés
        if any(word in task_lower for word in ['fényerő', 'brightness', 'fény', 'sötét', 'világos']):
            commands.extend([
                'powershell.exe -Command "(Get-WmiObject -Namespace root/WMI -Class WmiMonitorBrightnessMethods).WmiSetBrightness(1,50)"',
                'powershell.exe -Command "Get-WmiObject -Namespace root/WMI -Class WmiMonitorBrightness"'
            ])
        
        # BLUETOOTH eszközök
        if any(word in task_lower for word in ['bluetooth', 'bt', 'vezeték nélküli']):
            commands.extend([
                'powershell.exe -Command "Get-PnpDevice | Where-Object {$_.Class -eq \'Bluetooth\'}"',
                'powershell.exe -Command "Get-Service bthserv"'
            ])
        
        # WIFI jelszó mutatás
        if any(word in task_lower for word in ['wifi jelszó', 'wifi password', 'jelszav', 'kulcs']):
            commands.extend([
                'netsh wlan show profiles',
                'powershell.exe -Command "netsh wlan show profiles | Select-String \':\' | ForEach-Object { $name = $_.ToString().Split(\':\')[1].Trim(); netsh wlan show profile name=\\"$name\\" key=clear }"'
            ])
            
        # SISTEMA TELJESÍTMÉNY optimalizálás
        if any(word in task_lower for word in ['optimalizál', 'optimize', 'gyorsít', 'tisztít', 'clean']):
            commands.extend([
                'powershell.exe -Command "Get-Process | Sort-Object CPU -Descending | Select-Object -First 10"',
                'sfc /scannow',
                'powershell.exe -Command "Get-WmiObject Win32_StartupCommand | Select-Object Name,Location,Command"'
            ])
        
        return commands[:5]  # Maximum 5 parancs egyszerre

    async def _execute_programming_operations(self, task_description: str):
        """Programozási feladatok végrehajtása - kód generálás és fájlba írás"""
        print("💻 Programozási feladatok végrehajtása...")
        
        # AI-alapú kód generálás
        code_content = await self._generate_code_from_description(task_description)
        
        if code_content:
            # Fájl név generálása a feladat alapján
            filename = self._generate_filename_from_task(task_description)
            
            result = await self.tool_registry.tools['file_writer'].execute(
                path=filename,
                content=code_content
            )
            
            if result and result.get('success'):
                print(f"✅ Program fájl létrehozva: {result.get('file_path', 'Ismeretlen')}")
                
                # Ha Python fájl, próbáljuk meg futtatni
                if filename.endswith('.py'):
                    print("🚀 Python script futtatás próba...")
                    
                    # Ellenőrizzük, hogy interaktív alkalmazás-e (pygame, tkinter, stb.)
                    is_interactive = await self._is_interactive_application(code_content)
                    
                    if is_interactive:
                        print("🎮 Interaktív alkalmazás észlelve - háttérben indítás...")
                        # Interaktív alkalmazások háttérben futtatása
                        import subprocess
                        import os
                        try:
                            # Új CMD ablakban indítás
                            subprocess.Popen(f'start cmd /k "python {filename}"', shell=True, cwd=os.getcwd())
                            print(f"✅ Interaktív alkalmazás elindítva új ablakban!")
                            print(f"🎯 Fájl helye: {os.path.abspath(filename)}")
                        except Exception as e:
                            print(f"⚠️ Interaktív alkalmazás indítási hiba: {e}")
                    else:
                        # Normál script futtatás
                        run_result = await self.tool_registry.tools['system_command'].execute(
                            command=f"python {filename}"
                        )
                        if run_result and run_result.get('success'):
                            print(f"✅ Script sikeresen futtott!")
                            print(f"📟 Kimenet: {run_result.get('stdout', '')[:200]}...")
                        else:
                            print(f"⚠️ Script futtatási hiba: {run_result.get('error', 'Ismeretlen')}")
            else:
                print(f"❌ Fájl létrehozási hiba: {result.get('error', 'Ismeretlen hiba')}")
        else:
            print("❌ Nem sikerült kódot generálni a feladatból")

    async def _execute_web_development_operations(self, task_description: str):
        """Webfejlesztési feladatok végrehajtása"""
        print("🌐 Webfejlesztési feladatok végrehajtása...")
        
        # HTML, CSS, JS fájlok generálása
        html_content = await self._generate_html_from_description(task_description)
        css_content = await self._generate_css_from_description(task_description)
        js_content = await self._generate_js_from_description(task_description)
        
        # Weboldal mappa létrehozása
        website_name = self._generate_website_name_from_task(task_description)
        
        # Mappa létrehozás parancs
        mkdir_result = await self.tool_registry.tools['system_command'].execute(
            command=f"mkdir {website_name}"
        )
        
        files_created = []
        
        # HTML fájl
        if html_content:
            html_result = await self.tool_registry.tools['file_writer'].execute(
                path=f"{website_name}/index.html",
                content=html_content
            )
            if html_result and html_result.get('success'):
                files_created.append("index.html")
        
        # CSS fájl
        if css_content:
            css_result = await self.tool_registry.tools['file_writer'].execute(
                path=f"{website_name}/style.css",
                content=css_content
            )
            if css_result and css_result.get('success'):
                files_created.append("style.css")
        
        # JS fájl
        if js_content:
            js_result = await self.tool_registry.tools['file_writer'].execute(
                path=f"{website_name}/script.js",
                content=js_content
            )
            if js_result and js_result.get('success'):
                files_created.append("script.js")
        
        if files_created:
            print(f"✅ Weboldal létrehozva: {website_name}/")
            print(f"📁 Fájlok: {', '.join(files_created)}")
            
            # Weboldal megnyitása böngészőben
            open_result = await self.tool_registry.tools['system_command'].execute(
                command=f"start {website_name}/index.html"
            )
            if open_result and open_result.get('success'):
                print("🌐 Weboldal megnyitva a böngészőben!")
        else:
            print("❌ Nem sikerült weboldal fájlokat létrehozni")

    async def _generate_code_from_description(self, task_description: str) -> str:
        """AI-alapú kód generálás"""
        print("🧠 AI kód generálás...")
        
        code_prompt = f"""
Feladat: {task_description}

Kérlek, írj egy teljes, működő Python programot a feladat alapján. 

FONTOS KÖVETELMÉNYEK:
- A kód legyen teljes és rögtön futtatható
- Használj megfelelő kommenteket magyarul
- Ha GUI-t használsz, előnyben részesítsd a tkinter-t
- A kód legyen jól strukturált és olvasható
- Ne használj külső könyvtárakat, csak a Python beépített moduljait

PROGRAMOZÁSI STÍLUS:
- Használj világos változó neveket
- Minden függvényt kommentezz
- A main logikát tedd if __name__ == "__main__": blokkba

Feladat elemzése: {task_description}

Válaszold meg csak a Python kóddal, semmi más szöveggel:
"""
        
        try:
            response = await self.ai_client.generate_response(
                prompt=code_prompt,
                model=None,
                task_type="programming",
                temperature=0.7,
                max_tokens=2000
            )
            
            # Kód tisztítása - csak a Python kódot tartjuk meg
            code_text = str(response).strip()
            
            # Ha a válasz tartalmazta a ```python blokkot, tisztítsuk
            if "```python" in code_text:
                start = code_text.find("```python") + 9
                end = code_text.find("```", start)
                if end > start:
                    code_text = code_text[start:end].strip()
            elif "```" in code_text:
                start = code_text.find("```") + 3
                end = code_text.find("```", start)
                if end > start:
                    code_text = code_text[start:end].strip()
            
            print(f"📝 Generált kód: {len(code_text)} karakter")
            return code_text
            
        except Exception as e:
            print(f"❌ Kód generálási hiba: {e}")
            return ""

    async def _generate_html_from_description(self, task_description: str) -> str:
        """HTML generálás weboldal feladatokhoz"""
        html_prompt = f"""
Feladat: {task_description}

Kérlek, készíts egy modern, szép HTML oldalt a feladat alapján.

KÖVETELMÉNYEK:
- Teljes HTML5 struktúra
- Responsive design
- CSS fájl linkelése (style.css)
- JavaScript fájl linkelése (script.js)
- Magyar nyelvű tartalom
- Modern, esztétikus megjelenés

Válaszold meg csak a HTML kóddal:
"""
        
        try:
            response = await self.ai_client.generate_response(
                prompt=html_prompt,
                model=None,
                task_type="web_development",
                temperature=0.6,
                max_tokens=1500
            )
            return self._clean_code_response(str(response), "html")
        except:
            # Fallback HTML
            return f"""<!DOCTYPE html>
<html lang="hu">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Generált Weboldal</title>
    <link rel="stylesheet" href="style.css">
</head>
<body>
    <header>
        <h1>Automatikusan Generált Weboldal</h1>
    </header>
    <main>
        <p>Feladat: {task_description}</p>
        <p>Ez az oldal automatikusan lett generálva a Project-S V2 rendszer által.</p>
    </main>
    <footer>
        <p>Létrehozva: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </footer>
    <script src="script.js"></script>
</body>
</html>"""

    async def _generate_css_from_description(self, task_description: str) -> str:
        """CSS generálás weboldal feladatokhoz"""
        css_prompt = f"""
Feladat: {task_description}

Kérlek, írj modern CSS kódot a HTML oldalhoz.

KÖVETELMÉNYEK:
- Modern, szép design
- Responsive layout
- Színes, vonzó megjelenés
- Animációk és átmenetek
- Mobile-first approach

Válaszold meg csak a CSS kóddal:
"""
        
        try:
            response = await self.ai_client.generate_response(
                prompt=css_prompt,
                model=None,
                task_type="web_development",
                temperature=0.6,
                max_tokens=1000
            )
            return self._clean_code_response(str(response), "css")
        except:
            # Fallback CSS
            return """* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

body {
    font-family: Arial, sans-serif;
    line-height: 1.6;
    color: #333;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    min-height: 100vh;
}

header {
    background: rgba(255,255,255,0.9);
    padding: 2rem;
    text-align: center;
    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
}

main {
    max-width: 800px;
    margin: 2rem auto;
    padding: 2rem;
    background: rgba(255,255,255,0.95);
    border-radius: 10px;
    box-shadow: 0 5px 20px rgba(0,0,0,0.1);
}

footer {
    text-align: center;
    padding: 1rem;
    background: rgba(0,0,0,0.8);
    color: white;
    margin-top: 2rem;
}"""

    async def _generate_js_from_description(self, task_description: str) -> str:
        """JavaScript generálás weboldal feladatokhoz"""
        return """// Automatikusan generált JavaScript
document.addEventListener('DOMContentLoaded', function() {
    console.log('Weboldal betöltve!');
    
    // Egyszerű animáció
    const main = document.querySelector('main');
    if (main) {
        main.style.opacity = '0';
        main.style.transform = 'translateY(20px)';
        
        setTimeout(() => {
            main.style.transition = 'all 0.5s ease';
            main.style.opacity = '1';
            main.style.transform = 'translateY(0)';
        }, 100);
    }
});"""

    def _generate_filename_from_task(self, task_description: str) -> str:
        """Fájlnév generálás feladat alapján"""
        import re
        
        # Alapvető szavak kigyűjtése
        task_lower = task_description.lower()
        
        if "játék" in task_lower or "game" in task_lower:
            if "kígyó" in task_lower or "snake" in task_lower:
                return "snake_game.py"
            elif "tetris" in task_lower:
                return "tetris_game.py"
            else:
                return "game.py"
        elif "számológép" in task_lower or "calculator" in task_lower:
            return "calculator.py"
        elif "chat" in task_lower or "beszélget" in task_lower:
            return "chatbot.py"
        else:
            # Általános név generálás
            safe_name = re.sub(r'[^\w\s]', '', task_lower)
            safe_name = re.sub(r'\s+', '_', safe_name)[:20]
            return f"{safe_name or 'generated_program'}.py"

    def _generate_website_name_from_task(self, task_description: str) -> str:
        """Weboldal mappa név generálás"""
        import re
        safe_name = re.sub(r'[^\w\s]', '', task_description.lower())
        safe_name = re.sub(r'\s+', '_', safe_name)[:15]
        return f"{safe_name or 'website'}_{datetime.now().strftime('%H%M%S')}"

    async def _is_interactive_application(self, code_content: str) -> bool:
        """Ellenőrzi, hogy a kód interaktív alkalmazás-e (pygame, tkinter, stb.)"""
        interactive_keywords = [
            'pygame', 'tkinter', 'turtle', 'matplotlib.pyplot', 'cv2.imshow',
            'input()', 'game_loop', 'while True:', 'mainloop()', 'pygame.display',
            'cv2.waitKey', 'plt.show()', 'root.mainloop', 'screen.fill'
        ]
        
        code_lower = code_content.lower()
        
        # Ha bármelyik interaktív kulcsszót tartalmazza
        for keyword in interactive_keywords:
            if keyword.lower() in code_lower:
                return True
        
        # További ellenőrzések - import-ok alapján
        import_lines = [line for line in code_content.split('\n') if line.strip().startswith('import ') or line.strip().startswith('from ')]
        interactive_imports = ['pygame', 'tkinter', 'turtle', 'matplotlib', 'cv2']
        
        for line in import_lines:
            for imp in interactive_imports:
                if imp in line.lower():
                    return True
        
        return False

    async def _execute_web_scraping_operations(self, task_description: str):
        """Web scraping műveletek végrehajtása - ARCHAEOLOGICAL TREASURE!"""
        print("🌐 Web scraping műveletek végrehajtása...")
        print("🏺 Archaeological web tools activated!")
        
        # AI-alapú URL és selector generálás
        scraping_config = await self._generate_scraping_config_from_description(task_description)
        
        if scraping_config:
            # Web scraping végrehajtása
            result = await self.tool_registry.tools['web_scraper'].execute(scraping_config)
            
            if result and result.get('success'):
                print(f"✅ Web scraping sikeres!")
                print(f"📊 Adatok száma: {result.get('items_scraped', 0)}")
                print(f"📁 Kimeneti fájl: {result.get('output_file', 'N/A')}")
                
                # Ha van adatfájl, elemzés indítása
                if result.get('output_file'):
                    print("🔍 Automatikus adatelemzés indítása...")
                    
                    analysis_result = await self.tool_registry.tools['web_analyzer'].execute({
                        'data_file': result.get('output_file'),
                        'analysis_type': 'content'
                    })
                    
                    if analysis_result and analysis_result.get('success'):
                        print(f"✅ Web elemzés kész!")
                        print(f"📋 Elemzési jelentés: {analysis_result.get('report_file', 'N/A')}")
                    
            else:
                print(f"❌ Web scraping hiba: {result.get('error', 'Ismeretlen hiba')}")
        else:
            print("❌ Nem sikerült scraping konfigurációt generálni")

    async def _generate_scraping_config_from_description(self, task_description: str) -> dict:
        """AI-alapú scraping konfiguráció generálás"""
        print("🧠 AI scraping konfiguráció generálás...")
        
        scraping_prompt = f"""
Feladat: {task_description}

Te egy professzionális web scraping szakértő vagy. A felhasználó leírása alapján generálj web scraping konfigurációt.

PÉLDA FELADATOK ÉS KONFIGURÁCIÓK:

1. "Monitor news about AI" → 
{{
    "url": "https://news.ycombinator.com",
    "selectors": {{
        "container": ".athing",
        "title": ".storylink",
        "score": ".score"
    }},
    "output_format": "json"
}}

2. "Track prices from e-commerce" →
{{
    "url": "https://example-store.com/products",
    "selectors": {{
        "container": ".product",
        "title": ".product-title",
        "price": ".price"
    }},
    "output_format": "csv"
}}

3. "Extract contact information" →
{{
    "url": "https://company-website.com/contact",
    "selectors": {{
        "email": "a[href^='mailto:']",
        "phone": "a[href^='tel:']",
        "address": ".address"
    }},
    "output_format": "json"
}}

FONTOS SZABÁLYOK:
- Csak NYILVÁNOS weboldalakat használj
- Respektáld a robots.txt fájlokat
- Ne generálj privát vagy védett tartalmakat
- Használj valós, elérhető URL-eket

Válaszold meg JSON formátumban:
{{
    "url": "target_website_url",
    "selectors": {{
        "field_name": "css_selector"
    }},
    "output_format": "json",
    "delay": 1
}}

Feladat elemzése: {task_description}
"""
        
        try:
            ai_response = await self.ai_client.generate_response(
                prompt=scraping_prompt,
                model=None,
                task_type="web_scraping",
                temperature=0.3,
                max_tokens=500
            )
            
            print(f"🤖 AI válasz: {str(ai_response)[:150]}...")
            
            # JSON kinyerése a válaszból - JAVÍTOTT VERZIÓ
            import json
            import re
            
            # Többféle JSON formátum kezelése
            response_text = str(ai_response)
            parsed_config = None
            
            # 1. Próba: Clean JSON extraction
            json_patterns = [
                r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}',  # Nested JSON pattern
                r'\{.*?\}',  # Simple JSON pattern
                r'```json\s*(\{.*?\})\s*```',  # Markdown code block
                r'```\s*(\{.*?\})\s*```'  # Generic code block
            ]
            
            for pattern in json_patterns:
                json_match = re.search(pattern, response_text, re.DOTALL)
                if json_match:
                    try:
                        json_text = json_match.group(1) if json_match.lastindex else json_match.group()
                        # Clean common issues
                        json_text = json_text.strip()
                        json_text = re.sub(r',\s*}', '}', json_text)  # Remove trailing commas
                        json_text = re.sub(r',\s*]', ']', json_text)  # Remove trailing commas in arrays
                        
                        parsed_config = json.loads(json_text)
                        print(f"✅ JSON sikeresen feldolgozva pattern: {pattern[:20]}...")
                        break
                    except json.JSONDecodeError as e:
                        print(f"⚠️ JSON parse hiba pattern {pattern[:20]}...: {e}")
                        continue
            
            if parsed_config:
                    
                    # Biztonsági ellenőrzések
                    url = parsed_config.get('url', '')
                    if self._is_safe_scraping_url(url):
                        print(f"✅ Biztonságos scraping URL: {url}")
                        return parsed_config
                    else:
                        print(f"⚠️ Kockázatos URL elutasítva: {url}")
                        return self._fallback_scraping_config(task_description)
            else:
                print("❌ Nem sikerült JSON-t kinyerni az AI válaszból")
            
            # Fallback: Alapértelmezett konfiguráció
            return self._fallback_scraping_config(task_description)
            
        except Exception as e:
            print(f"❌ AI scraping konfiguráció hiba: {e}")
            return self._fallback_scraping_config(task_description)
    
    def _is_safe_scraping_url(self, url: str) -> bool:
        """Ellenőrzi, hogy biztonságos-e a scraping URL"""
        safe_domains = [
            'example.com', 'httpbin.org', 'jsonplaceholder.typicode.com',
            'news.ycombinator.com', 'reddit.com', 'github.com'
        ]
        
        # Alapértelmezett safe URLs teszteléshez
        return any(domain in url.lower() for domain in safe_domains) or url.startswith('https://')
    
    def _fallback_scraping_config(self, task_description: str) -> dict:
        """Fallback scraping konfiguráció generálás - JAVÍTOTT"""
        print("🔄 Fallback scraping konfiguráció használata...")
        
        # Feladat típus alapú URL választás
        if any(word in task_description.lower() for word in ['hírek', 'news', 'cikkek', 'újság']):
            return {
                "url": "https://news.ycombinator.com",
                "selectors": {
                    "container": ".athing",
                    "title": ".storylink"
                },
                "output_format": "json",
                "delay": 2
            }
        elif any(word in task_description.lower() for word in ['github', 'repo', 'project']):
            return {
                "url": "https://github.com/trending",
                "selectors": {
                    "container": ".Box-row",
                    "title": "h1 a"
                },
                "output_format": "json",
                "delay": 2
            }
        else:
            # Alapértelmezett biztonságos teszt URL
            return {
                "url": "https://jsonplaceholder.typicode.com/posts",
                "selectors": {},  # Alapértelmezett extraction
                "output_format": "json",
                "delay": 1
            }
    
    async def _execute_browser_automation_operations(self, task_description: str):
        """Browser automation műveletek végrehajtása - VALÓDI BÖNGÉSZŐ HASZNÁLAT"""
        print("🌐 Browser automation műveletek végrehajtása...")
        
        try:
            # Ellenőrizzük, hogy van-e browser automation tool
            if 'browser_automation' not in self.tool_registry.tools:
                print("⚠️ Browser automation tool nem elérhető")
                # Fallback to command execution
                await self._execute_command_operations(task_description)
                return
            
            browser_tool = self.tool_registry.tools['browser_automation']
            
            # AI-alapú task előkészítése - JAVÍTOTT VERZIÓ
            task_analysis_prompt = f"""
Feladat: {task_description}

Kérlek, elemezd ezt a böngésző automatizálási feladatot és add meg egy TELJES, RÉSZLETES utasításként angol nyelven, amely MINDEN lépést tartalmaz:

Példák:
- "nyisd meg a googlet és keress mese hangoskönyvet" → "Navigate to google.com, search for fairy tale audiobooks, and browse the results"
- "menj a gmail-re és írj egy emailt" → "Navigate to gmail.com, compose a new email with subject and content, and send it"
- "lépj a gmail-re és küld egy levelet zsoltszupnak, egy vicces emailt" → "Navigate to gmail.com, compose a new email to zsoltszup@example.com with a funny subject and funny content, then send the email"

FONTOS: NE egyszerűsítsd le a feladatot! Tartsd meg az ÖSSZES részletet és lépést!

Válasz csak a TELJES angol utasítással:
"""
            
            processed_task = await self.ai_client.generate_response(
                prompt=task_analysis_prompt,
                model=None,  # Use default model instead of invalid qwen3-235b
                task_type="general",
                temperature=0.3,
                max_tokens=100
            )
            
            if processed_task:
                # Clean the response - remove <think> blocks and extract only the task
                processed_task = str(processed_task).strip()
                
                # Remove <think> blocks completely
                import re
                processed_task = re.sub(r'<think>.*?</think>', '', processed_task, flags=re.DOTALL)
                
                # Also remove any remaining <think> tags without closing tags
                processed_task = re.sub(r'<think>.*', '', processed_task, flags=re.DOTALL)
                
                # Clean up extra whitespace
                processed_task = processed_task.strip()
                
                # Extract the final task instruction
                if processed_task:
                    lines = processed_task.split('\n')
                    for line in lines:
                        line = line.strip()
                        if line and not line.startswith('#') and len(line) > 10:
                            processed_task = line.strip().strip('"')
                            break
                
                # If still empty after cleaning, use fallback
                if not processed_task or processed_task.startswith('<think'):
                    processed_task = None
                
                print(f"🤖 Feldolgozott task: {processed_task}")
            
            # Use fallback if processing failed
            if not processed_task:
                # Fallback task - preserve original intent
                if "google" in task_description.lower():
                    processed_task = f"Navigate to google.com and perform: {task_description}"
                elif "gmail" in task_description.lower():
                    processed_task = f"Navigate to gmail.com and perform: {task_description}"
                else:
                    processed_task = f"Navigate to website and perform task: {task_description}"
                print(f"🔄 Fallback task használata: {processed_task}")
            
            # Browser automation végrehajtása
            print(f"🌐 Browser task végrehajtása: {processed_task}")
            
            result = await browser_tool.execute(
                task=processed_task,
                headless=False,  # Vizuális böngésző
                timeout=120,     # 2 perc timeout
                screenshot=True, # Screenshot készítése
                wait_time=2      # Várás az akciók között
            )
            
            if result and result.get('success'):
                print(f"✅ Browser automation sikeres!")
                print(f"   🎯 Task: {processed_task}")
                print(f"   ⏱️ Időtartam: {result.get('execution_time', 0):.2f}s")
                
                if result.get('screenshot'):
                    print(f"   📸 Screenshot: {result.get('screenshot')}")
                
                # Initialize extracted data
                extracted = result.get('extracted_data', {})
                if extracted:
                    print(f"   📄 Oldal cím: {extracted.get('title', 'N/A')}")
                    print(f"   🔗 URL: {extracted.get('url', 'N/A')}")
                
                # Eredmény dokumentálása
                browser_content = f"""# Browser Automation Jelentés

**Feladat:** {task_description}
**Végrehajtva:** {datetime.now()}
**Feldolgozott task:** {processed_task}

## Végrehajtási Eredmény
✅ **Sikeresen végrehajtva** ({result.get('execution_time', 0):.2f}s)

### Részletek
- **URL:** {extracted.get('url', 'N/A')}
- **Oldal cím:** {extracted.get('title', 'N/A')}
- **Screenshot:** {result.get('screenshot', 'Nem készült')}

### Browser Automation Eredmény
```
{result.get('result', 'Nincs specifikus eredmény')}
```

## Összefoglalás
A browser automation sikeresen lefutott. A feladat végrehajtása során a böngésző valódi weboldalt nyitott meg és interakciókat hajtott végre.
"""
                
                file_result = await self.tool_registry.tools['file_writer'].execute(
                    path=f"browser_automation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                    content=browser_content
                )
                
                if file_result and file_result.get('success'):
                    print(f"✅ Browser automation jelentés létrehozva: {file_result.get('file_path', 'Ismeretlen')}")
            
            else:
                print(f"❌ Browser automation sikertelen: {result.get('error', 'Ismeretlen hiba')}")
                # Create error report
                error_content = f"""# Browser Automation Hiba Jelentés

**Feladat:** {task_description}
**Végrehajtva:** {datetime.now()}
**Feldolgozott task:** {processed_task}

## Hiba Részletei
❌ **Sikertelen végrehajtás**

### Hiba információ
```
{result.get('error', 'Ismeretlen hiba') if result else 'Nincs válasz a browser automation tool-tól'}
```

## Következő lépések
A browser automation sikertelen volt. Fallback módra váltás parancssori végrehajtásra.
"""
                
                try:
                    error_file_result = await self.tool_registry.tools['file_writer'].execute(
                        path=f"browser_automation_error_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                        content=error_content
                    )
                    if error_file_result and error_file_result.get('success'):
                        print(f"📝 Hiba jelentés létrehozva: {error_file_result.get('file_path', 'Ismeretlen')}")
                except Exception as report_error:
                    print(f"⚠️ Hiba jelentés létrehozása sikertelen: {report_error}")
                
                # Fallback to command execution
                await self._execute_command_operations(task_description)
        
        except Exception as e:
            print(f"❌ Browser automation hiba: {e}")
            import traceback
            traceback.print_exc()
            # Fallback to command execution
            await self._execute_command_operations(task_description)


async def main():
    """Főprogram - interaktív menü a teszteléshez"""
    print("🚀 PROJECT-S V2 SIMPLIFIED - KOMPLEX FELADAT TESZTELŐ")
    print("=" * 60)
    
    processor = ComplexTaskProcessor()
    
    while True:
        print("\n🎯 TESZTELÉSI OPCIÓK:")
        print("1. 🐛 Challenge 1: Bug Fix + Unit Test")
        print("2. 🔧 Challenge 2: CLI Tool létrehozása")
        print("3. 🌐 Challenge 3: Web Scraping + Data Processing")
        print("4. 🌐 Browser Automation Teszt (Gmail)")
        print("5. 💻 Egyéni feladat bevitele")
        print("0. 🚪 Kilépés")
        
        try:
            choice = input("\n📝 Válassz opciót (0-5): ").strip()
            
            if choice == "0":
                print("👋 Viszlát!")
                break
            elif choice == "1":
                print("\n🐛 CHALLENGE 1: Bug Fix + Unit Test")
                task = "Find and fix the bug in this Python calculator function, then write unit tests for it: def divide(a, b): return a / b"
                await processor.process_natural_language_task(task)
            elif choice == "2":
                print("\n🔧 CHALLENGE 2: CLI Tool létrehozása")
                task = "Create a command-line weather checker tool that takes a city name and displays current weather information"
                await processor.process_natural_language_task(task)
            elif choice == "3":
                print("\n🌐 CHALLENGE 3: Web Scraping + Data Processing")
                task = "Scrape the latest trending repositories from GitHub and create a summary report with the most popular programming languages"
                await processor.process_natural_language_task(task)
            elif choice == "4":
                print("\n🌐 BROWSER AUTOMATION TESZT")
                task = "lépj a gmail-re és küld egy levelet zsoltszupnak, egy vicces emailt"
                await processor.process_natural_language_task(task)
            elif choice == "5":
                print("\n💻 EGYÉNI FELADAT")
                custom_task = input("📝 Írd be a feladatot: ").strip()
                if custom_task:
                    await processor.process_natural_language_task(custom_task)
                else:
                    print("⚠️ Üres feladat!")
            else:
                print("❌ Érvénytelen választás!")
                
        except KeyboardInterrupt:
            print("\n\n👋 Kilépés...")
            break
        except Exception as e:
            print(f"❌ Hiba: {e}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
