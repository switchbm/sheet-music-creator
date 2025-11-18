### Type Checking Issues Found

**Total Errors: 15 across 4 files**

#### app/services/quantizer.py (2 errors)
- Line 111: `min_duration: float = None` should be `Optional[float] = None`
- Line 168: `resolution: int = None` should be `Optional[int] = None`

#### app/utils/audio_loader.py (3 errors)
- Line 49: Type mismatch - Path object assigned to str variable
- Line 50: Calling .exists() on str instead of Path
- Line 67: Return type mismatch - librosa.load can return float or int

#### app/services/engine.py (9 errors)
- Line 36: `device: str = None` should be `Optional[str] = None`
- Lines 52, 54, 69, 105, 116, 165, 177: Cannot determine type of "device" (cascading from line 36)

#### main.py (1 error)
- Line 140: file.filename can be None, not checked before validation
- Line 163: file.filename can be None, not checked before use

**STATUS: NOT MYPY COMPLIANT**
