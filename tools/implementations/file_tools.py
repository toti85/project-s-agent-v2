"""
Project-S File Tools
------------------
Ez a modul a fájlműveletekhez kapcsolódó eszközöket tartalmazza:
- Fájl olvasás és írás
- Fájl keresés és listázás
- Fájl információk
"""

import os
import asyncio
import aiofiles
import glob
import logging
import json
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import shutil
import fnmatch

from tools.tool_interface import BaseTool
from tools.tool_registry import tool_registry

logger = logging.getLogger(__name__)

class FileReadTool(BaseTool):
    """
    Fájl tartalmának olvasása és visszaadása.
    
    Category: file
    Version: 1.0.0
    Requires permissions: Yes
    Safe: Yes
    """
    
    async def execute(self, 
                    path: str, 
                    encoding: str = 'utf-8',
                    max_size: int = 1024 * 1024) -> Dict[str, Any]:
        """
        Beolvassa egy fájl tartalmát.
        
        Args:
            path: A fájl elérési útja
            encoding: A fájl karakterkódolása
            max_size: Maximális olvasható méret bájtban (alapértelmezett: 1MB)
            
        Returns:
            Dict: Az eredmény szótár formában
        """
        try:
            # Biztonsági ellenőrzés
            security_check = tool_registry.check_security("file_read", {"path": path})
            if not security_check["allowed"]:
                return {
                    "success": False,
                    "error": security_check["reason"]
                }
            
            # Fájl elérési út normalizálása
            file_path = Path(path).resolve()
            
            # Ellenőrizzük, hogy létezik-e a fájl
            if not file_path.exists():
                return {
                    "success": False,
                    "error": f"A fájl nem létezik: {path}"
                }
                
            # Ellenőrizzük, hogy tényleg fájl-e
            if not file_path.is_file():
                return {
                    "success": False,
                    "error": f"Az elérési út nem egy fájlra mutat: {path}"
                }
                
            # Ellenőrizzük a fájl méretét
            file_size = file_path.stat().st_size
            if file_size > max_size:
                return {
                    "success": False,
                    "error": f"A fájl mérete ({file_size} bájt) meghaladja a megengedett méretet ({max_size} bájt)"
                }
                
            # Fájl olvasása
            async with aiofiles.open(file_path, 'r', encoding=encoding) as f:
                content = await f.read()
                
            return {
                "success": True,
                "content": content,
                "size": len(content),
                "encoding": encoding,
                "path": str(file_path)
            }
        
        except UnicodeDecodeError:
            return {
                "success": False,
                "error": f"Kódolási hiba: a fájl nem olvasható {encoding} kódolással"
            }
        except Exception as e:
            logger.error(f"Hiba a fájl olvasása közben: {str(e)}")
            return {
                "success": False,
                "error": f"Hiba történt: {str(e)}"
            }


class FileWriteTool(BaseTool):
    """
    Tartalom írása fájlba.
    
    Category: file
    Version: 1.0.0
    Requires permissions: Yes
    Safe: No
    """
    
    async def execute(self, 
                    path: str, 
                    content: str, 
                    encoding: str = 'utf-8',
                    mode: str = 'w') -> Dict[str, Any]:
        """
        Tartalmat ír egy fájlba.
        
        Args:
            path: A fájl elérési útja
            content: A fájlba írandó tartalom
            encoding: A fájl karakterkódolása
            mode: Írási mód ('w' - felülírás, 'a' - hozzáfűzés)
            
        Returns:
            Dict: Az eredmény szótár formában
        """
        try:
            # Ellenőrizzük az írási módot
            if mode not in ['w', 'a']:
                return {
                    "success": False,
                    "error": f"Érvénytelen írási mód: {mode} (csak 'w' vagy 'a' engedélyezett)"
                }
            
            # Biztonsági ellenőrzés
            security_check = tool_registry.check_security("file_write", {
                "path": path,
                "size": len(content.encode(encoding))
            })
            if not security_check["allowed"]:
                return {
                    "success": False,
                    "error": security_check["reason"]
                }
                
            # Fájl elérési út normalizálása
            file_path = Path(path).resolve()
            
            # Létrehozzuk a könyvtárat, ha nem létezik
            os.makedirs(file_path.parent, exist_ok=True)
            
            # Fájl írása
            async with aiofiles.open(file_path, mode, encoding=encoding) as f:
                await f.write(content)
                
            return {
                "success": True,
                "path": str(file_path),
                "size": len(content.encode(encoding)),
                "mode": mode
            }
            
        except Exception as e:
            logger.error(f"Hiba a fájl írása közben: {str(e)}")
            return {
                "success": False,
                "error": f"Hiba történt: {str(e)}"
            }


class FileSearchTool(BaseTool):
    """
    Fájlok keresése minta alapján.
    
    Category: file
    Version: 1.0.0
    Requires permissions: Yes
    Safe: Yes
    """
    
    async def execute(self, 
                    pattern: str, 
                    root_dir: Optional[str] = None,
                    recursive: bool = True,
                    max_results: int = 100) -> Dict[str, Any]:
        """
        Fájlokat keres minta alapján.
        
        Args:
            pattern: Keresési minta (pl. "*.py", "config.*")
            root_dir: A kezdő könyvtár (alapértelmezetten a jelenlegi)
            recursive: Rekurzív keresés alkönyvtárakban
            max_results: Maximálisan visszaadott találatok száma
            
        Returns:
            Dict: A találatok listája
        """
        try:
            # Az alapértelmezett könyvtár beállítása
            if root_dir is None:
                root_dir = os.getcwd()
                
            # Biztonsági ellenőrzés
            security_check = tool_registry.check_security("file_read", {"path": root_dir})
            if not security_check["allowed"]:
                return {
                    "success": False,
                    "error": security_check["reason"]
                }
            
            # Elérési út normalizálása
            search_root = Path(root_dir).resolve()
            
            # Ellenőrizzük, hogy létezik-e a könyvtár
            if not search_root.exists() or not search_root.is_dir():
                return {
                    "success": False,
                    "error": f"A megadott könyvtár nem létezik: {root_dir}"
                }
            
            # Fájlok keresése
            results = []
            
            if recursive:
                # Rekurzív keresés az összes alkönyvtárban
                for path in search_root.glob('**/' + pattern):
                    if len(results) >= max_results:
                        break
                    results.append({
                        "path": str(path),
                        "name": path.name,
                        "size": path.stat().st_size,
                        "is_dir": path.is_dir()
                    })
            else:
                # Csak az adott könyvtárban keresés
                for path in search_root.glob(pattern):
                    if len(results) >= max_results:
                        break
                    results.append({
                        "path": str(path),
                        "name": path.name,
                        "size": path.stat().st_size,
                        "is_dir": path.is_dir()
                    })
                    
            return {
                "success": True,
                "results": results,
                "count": len(results),
                "pattern": pattern,
                "root_dir": str(search_root),
                "has_more": len(results) >= max_results
            }
            
        except Exception as e:
            logger.error(f"Hiba a fájlkeresés közben: {str(e)}")
            return {
                "success": False,
                "error": f"Hiba történt: {str(e)}"
            }


class FileInfoTool(BaseTool):
    """
    Információk lekérése fájlokról vagy könyvtárakról.
    
    Category: file
    Version: 1.0.0
    Requires permissions: Yes
    Safe: Yes
    """
    
    async def execute(self, path: str) -> Dict[str, Any]:
        """
        Információkat ad egy fájlról vagy könyvtárról.
        
        Args:
            path: A fájl vagy könyvtár elérési útja
            
        Returns:
            Dict: Az eredmény szótár formában
        """
        try:
            # Biztonsági ellenőrzés
            security_check = tool_registry.check_security("file_read", {"path": path})
            if not security_check["allowed"]:
                return {
                    "success": False,
                    "error": security_check["reason"]
                }
            
            # Fájl elérési út normalizálása
            file_path = Path(path).resolve()
            
            # Ellenőrizzük, hogy létezik-e
            if not file_path.exists():
                return {
                    "success": False,
                    "error": f"A megadott útvonal nem létezik: {path}"
                }
                
            # Alap információk
            stat_info = file_path.stat()
            info = {
                "name": file_path.name,
                "path": str(file_path),
                "size": stat_info.st_size,
                "is_dir": file_path.is_dir(),
                "is_file": file_path.is_file(),
                "modified_time": stat_info.st_mtime,
                "created_time": stat_info.st_ctime
            }
            
            # További információk könyvtárak esetén
            if file_path.is_dir():
                info["contents"] = [child.name for child in file_path.iterdir()]
                info["files_count"] = len([1 for child in file_path.iterdir() if child.is_file()])
                info["dirs_count"] = len([1 for child in file_path.iterdir() if child.is_dir()])
            
            # Fájl kiterjesztés, ha fájl
            if file_path.is_file():
                info["extension"] = file_path.suffix
                
            return {
                "success": True,
                "info": info
            }
            
        except Exception as e:
            logger.error(f"Hiba a fájl információk lekérése közben: {str(e)}")
            return {
                "success": False,
                "error": f"Hiba történt: {str(e)}"
            }


class FileContentSearchTool(BaseTool):
    """
    Tartalom keresése fájlokban.
    
    Category: file
    Version: 1.0.0
    Requires permissions: Yes
    Safe: Yes
    """
    
    async def execute(self, 
                    search_text: str, 
                    file_pattern: str = "*.*",
                    root_dir: Optional[str] = None,
                    max_results: int = 50,
                    case_sensitive: bool = False) -> Dict[str, Any]:
        """
        Szöveget keres fájlok tartalmában.
        
        Args:
            search_text: A keresendő szöveg
            file_pattern: A vizsgálandó fájlok mintája (pl. "*.py")
            root_dir: A kezdő könyvtár (alapértelmezetten a jelenlegi)
            max_results: Maximálisan visszaadott találatok száma
            case_sensitive: Kis- és nagybetű érzékenység
            
        Returns:
            Dict: A találatok listája
        """
        try:
            # Az alapértelmezett könyvtár beállítása
            if root_dir is None:
                root_dir = os.getcwd()
                
            # Biztonsági ellenőrzés
            security_check = tool_registry.check_security("file_read", {"path": root_dir})
            if not security_check["allowed"]:
                return {
                    "success": False,
                    "error": security_check["reason"]
                }
            
            # Elérési út normalizálása
            search_root = Path(root_dir).resolve()
            
            # Ellenőrizzük, hogy létezik-e a könyvtár
            if not search_root.exists() or not search_root.is_dir():
                return {
                    "success": False,
                    "error": f"A megadott könyvtár nem létezik: {root_dir}"
                }
                
            results = []
            search_str = search_text if case_sensitive else search_text.lower()
            
            # Keresés minden illeszkedő fájlban
            for path in search_root.glob(f"**/{file_pattern}"):
                if not path.is_file():
                    continue
                    
                try:
                    # Fájl olvasása és keresés
                    async with aiofiles.open(path, 'r', encoding='utf-8', errors='ignore') as f:
                        line_number = 0
                        async for line in f:
                            line_number += 1
                            line_to_search = line if case_sensitive else line.lower()
                            
                            if search_str in line_to_search:
                                results.append({
                                    "path": str(path),
                                    "line_number": line_number,
                                    "line": line.strip(),
                                    "context": line.strip()
                                })
                                
                                if len(results) >= max_results:
                                    break
                                    
                    if len(results) >= max_results:
                        break
                        
                except Exception as e:
                    logger.warning(f"Nem sikerült olvasni a fájlt: {str(path)}: {str(e)}")
                    continue
                    
            return {
                "success": True,
                "results": results,
                "count": len(results),
                "search_text": search_text,
                "file_pattern": file_pattern,
                "has_more": len(results) >= max_results
            }
            
        except Exception as e:
            logger.error(f"Hiba a tartalomkeresés közben: {str(e)}")
            return {
                "success": False,
                "error": f"Hiba történt: {str(e)}"
            }


# Az eszközöket regisztráljuk a példányosításkor
file_read_tool = FileReadTool()
file_write_tool = FileWriteTool()
file_search_tool = FileSearchTool()
file_info_tool = FileInfoTool()
file_content_search_tool = FileContentSearchTool()

# Hozzáadjuk őket az exportált eszközökhöz
__all__ = ['file_read_tool', 'file_write_tool', 'file_search_tool', 
           'file_info_tool', 'file_content_search_tool']
