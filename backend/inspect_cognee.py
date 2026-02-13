import inspect
import os
import sys

try:
    import cognee
    print(f"Cognee Version: {cognee.__version__}")
    
    from cognee.tasks.storage import add_data_points
    print(f"add_data_points signature: {inspect.signature(add_data_points)}")
    
    from cognee.shared.utils import get_anonymous_id
    print(f"get_anonymous_id source file: {inspect.getfile(get_anonymous_id)}")
    
except Exception as e:
    print(f"Error: {e}")
