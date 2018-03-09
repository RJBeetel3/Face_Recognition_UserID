try: 
	from setuptools import setup
except ImportError:
	from distutils.core import setup
	
config = {
	'description': 'Face_Recognition_User_ID', 
	'author': 'Rob Beetel', 
	'url': 'Not yet available online', 
	'download_url': 'Not yet available online', 
	'author_email': 'beetel2@gmail.com', 
	'version': '0.1', 
	'install_requires': ['nose'], 
	'packages': ['Name'], 
	'scripts': [],
	'name': 'projectname'
	}

setup(**config)