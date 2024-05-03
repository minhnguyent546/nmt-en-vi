from pathlib import Path
import subprocess
from typing import Literal


class MosesDetokenizer(object):
    def __init__(
        self,
        lang: str = 'en',
        *,
        uppercase_first: bool = False,
        disable_perl_buffering: bool = False,
        penn: bool = False,
    ) -> None:
        scripts_dir = Path(__file__).absolute().parents[2] / 'scripts'
        moses_detokenizer = str(scripts_dir / 'moses/scripts/tokenizer/detokenizer.perl')

        self.command = [
            'perl', moses_detokenizer,
            '-l', lang,
        ]
        if uppercase_first:
            self.command.append('-u')
        if disable_perl_buffering:
            self.command.append('-b')
        if penn:
            self.command.append('-penn')

    def __call__(
        self,
        sentence: str | list[str],
        format: Literal['text'] | None = None
    ) -> str | list[str] | list[list[str]]:
        text_input = ''
        if isinstance(sentence, str):
            if '\n' in sentence:
                raise ValueError('Newline characters are not allowed in the sentence to be detokenized.')
            text_input = sentence.rstrip()
        else:
            for sent in sentence:
                if '\n' in sent:
                    raise ValueError('Newline characters are not allowed in the sentence to be detokenized.')

            text_input = '\n'.join(sent.rstrip() for sent in sentence)

        completed_process = subprocess.run(self.command, input=text_input.encode(), capture_output=True)
        tokenized_text = completed_process.stdout.decode().rstrip().split('\n')
        if format != 'text':
            tokenized_text = [sentence.split(' ') for sentence in tokenized_text]

        if len(tokenized_text) == 1:
            tokenized_text = tokenized_text[0]

        return tokenized_text
