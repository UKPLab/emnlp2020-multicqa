#
# Reads ids of duplicate pairs from the SE data dump and print them to a file (so that we can exclude them in the
# training data)
#

import io
import logging
import subprocess
import xml.etree.ElementTree as ET

import click
import tqdm
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('root')


class SELinkReader(object):
    """
    NOTE: Example file
        <?xml version="1.0" encoding="utf-8"?>
        <postlinks>
          <row Id="27" CreationDate="2013-12-18T16:21:09.357" PostId="65" RelatedPostId="82" LinkTypeId="1" />
          <row Id="155" CreationDate="2013-12-19T14:28:22.633" PostId="154" RelatedPostId="100" LinkTypeId="1" />
          ...
    """

    def __init__(self, data_file_path):
        """
        data_file_path : (string) path to the PostLinks.xml file
        """
        self.data_file_path = data_file_path

    def n_items_unfiltered(self):
        out = subprocess.check_output(['wc', '-l', self.data_file_path])
        return int(out.split()[0])

    def read_ids(self):
        ids = set()
        with io.open(self.data_file_path, 'r', encoding="utf-8") as f:
            for l in tqdm(f):
                try:
                    sample = ET.fromstring(l.strip()).attrib
                    # id=3 -> duplicate https://meta.stackexchange.com/questions/2677/database-schema-documentation-for-the-public-data-dump-and-sede
                    if sample['LinkTypeId'] == '3':
                        ids.add(sample['PostId'])
                        ids.add(sample['RelatedPostId'])
                except ET.ParseError as e:
                    logger.info('(Ignoring) ERROR in parsing line (QUESTION READER):\n{}\n'.format(l.strip()))
        return ids


@click.command()
@click.argument('se_input')
@click.argument('output', type=click.File('wt'))
def create_data(se_input, output):
    reader = SELinkReader(se_input)
    dup_ids = reader.read_ids()
    output.write('{}\n'.format('\n'.join(dup_ids)))


if __name__ == "__main__":
    create_data()
