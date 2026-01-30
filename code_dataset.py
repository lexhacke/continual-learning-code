import ast
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from google import genai
from dotenv import load_dotenv
import os
from torch.utils.data import Dataset

load_dotenv()

class CodeFinetuneDataset(Dataset):
    """
    A dataset that loads Python scripts from a GitHub repository and creates
    (context, target) pairs where context is the script with a class/function
    removed and target is the removed class/function.
    """

    def __init__(self, repo_url: str, local_path: Optional[str] = None, extensions: Tuple[str, ...] = ('.py',)):
        """
        Args:
            repo_url: GitHub repository URL to clone
            local_path: Optional local path to clone to. If None, uses a temp directory.
            extensions: File extensions to include (default: Python files only)
        """
        self.client = genai.Client(api_key=os.environ['GOOGLE_KEY'])
        self.repo_url = repo_url
        self.extensions = extensions

        if local_path:
            self.repo_path = Path(local_path)
        else:
            self.temp_dir = tempfile.TemporaryDirectory()
            self.repo_path = Path(self.temp_dir.name)

        self._clone_repo()
        self._load_scripts()
        self._build_samples()

    def _clone_repo(self):
        """Clone the GitHub repository."""
        if not (self.repo_path / '.git').exists():
            subprocess.run(
                ['git', 'clone', '--depth', '1', self.repo_url, str(self.repo_path)],
                check=True,
                capture_output=True
            )

    def _load_scripts(self):
        """Find and load all script files from the repository."""
        self.scripts: List[Tuple[Path, str]] = []

        for ext in self.extensions:
            for filepath in self.repo_path.rglob(f'*{ext}'):
                if '.git' in filepath.parts:
                    continue
                try:
                    content = filepath.read_text(encoding='utf-8')
                    self.scripts.append((filepath, content))
                except (UnicodeDecodeError, PermissionError):
                    continue

    def _build_samples(self):
        """
        Parse each script and build (context, target) samples.
        Each sample is a tuple of (script_with_hole, removed_block).
        """
        self.samples: List[Tuple[str, str, str]] = []  # (filepath, context, target)

        for filepath, content in self.scripts:
            try:
                tree = ast.parse(content)
            except SyntaxError:
                continue

            lines = content.splitlines(keepends=True)

            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    if not hasattr(node, 'lineno') or not hasattr(node, 'end_lineno'):
                        continue

                    start_line = node.lineno - 1
                    end_line = node.end_lineno

                    removed_block = ''.join(lines[start_line:end_line])

                    context_lines = lines[:start_line] + lines[end_line:]
                    context = ''.join(context_lines)

                    if removed_block.strip() and context.strip():
                        self.samples.append((
                            str(filepath.relative_to(self.repo_path)),
                            context,
                            removed_block
                        ))

    def _generate_preconditions(self, context: str, code: str) -> str:
        """Use Gemini to generate pre/post conditions for the code block."""
        prompt = f"""Given the following code context and a function/class that belongs in it,
generate a concise description of the pre-conditions and post-conditions for the code.

Context (surrounding code):
```python
{context[:2000]}
```

Code to describe:
```python
{code}
```

Provide a clear, concise specification describing:
1. What inputs/state this code expects (pre-conditions)
2. What outputs/state changes this code produces (post-conditions)
3. The purpose and behavior of the code

Be concise and precise."""

        response = self.client.models.generate_content(
            model='gemini-2.0-flash',
            contents=prompt
        )
        return response.text

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, str]:
        """
        Returns:
            Dict with:
                - "context": the script with the class/function removed (masked)
                - "prompt": Gemini-generated pre/post conditions
                - "code": the removed class/function
        """
        _, context, code = self.samples[idx]
        prompt = self._generate_preconditions(context, code)
        return {
            "context": context,
            "prompt": prompt,
            "code": code
        }


if __name__ == '__main__':
    dataset = CodeFinetuneDataset('https://github.com/facebookresearch/dinov3')
    print(f'Loaded {len(dataset)} samples')

    if len(dataset) > 0:
        item = dataset[0]
        context, prompt, code = item['context'], item['prompt'], item['code']
        print(f'\n--- Generated Prompt ---\n{prompt}\n')
        print(f'\n--- Context (script with hole) ---\n{context[:500]}...')
        print(f'\n--- Target (removed block) ---\n{code[:500]}...')
