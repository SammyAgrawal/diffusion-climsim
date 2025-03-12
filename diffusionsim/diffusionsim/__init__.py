__version__ = "0.1.0"
#import climsim_training_utils
DIFFUSERS_AVAILABLE = False
try:
    import diffusers
    DIFFUSERS_AVAILABLE = True
except ImportError:
    print("Diffusers not in workspace, not all models can be loaded")

# If you need to expose diffusers to other modules
if DIFFUSERS_AVAILABLE:
    __all__ = ['DIFFUSERS_AVAILABLE', 'diffusers']
else:
    __all__ = ['DIFFUSERS_AVAILABLE']