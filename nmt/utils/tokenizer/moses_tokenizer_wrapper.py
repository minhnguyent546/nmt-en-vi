from pathlib import Path
import subprocess
from typing import Literal


class MosesTokenizer(object):
    def __init__(
        self,
        lang: str = 'en',
        threads: int = 1,
        *,
        aggr_hyphen_splitting: bool = False,
        disable_perl_buffering: bool = False,
        time: bool = False,
        penn: bool = False,
        html_escape: bool = True,
    ) -> None:
        scripts_dir = Path(__file__).absolute().parents[2] / 'scripts'
        moses_tokenizer = str(scripts_dir / 'moses/scripts/tokenizer/tokenizer.perl')
        self.command = [
            'perl', moses_tokenizer,
            '-l', lang,
            '-threads', str(threads),
        ]
        if aggr_hyphen_splitting:
            self.command.append('-a')
        if disable_perl_buffering:
            self.command.append('-b')
        if time:
            self.command.append('-time')
        if penn:
            self.command.append('-penn')
        if not html_escape:
            self.command.append('-no-escape')

    def __call__(
        self,
        sentence: str | list[str],
        format: Literal['text'] | None = None
    ) -> str | list[str] | list[list[str]]:
        text_input = ''
        if isinstance(sentence, str):
            if '\n' in sentence:
                raise ValueError('Newline characters are not allowed in the sentence to be tokenized.')
            text_input = sentence.rstrip()
        else:
            for sent in sentence:
                if '\n' in sent:
                    raise ValueError('Newline characters are not allowed in the sentence to be tokenized.')

            text_input = '\n'.join(sent.rstrip() for sent in sentence)

        completed_process = subprocess.run(self.command, input=text_input.encode(), capture_output=True)
        tokenized_text = completed_process.stdout.decode().rstrip().split('\n')
        if format != 'text':
            tokenized_text = [sentence.split(' ') for sentence in tokenized_text]

        if len(tokenized_text) == 1:
            tokenized_text = tokenized_text[0]

        return tokenized_text
