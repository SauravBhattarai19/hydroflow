# -*- coding: utf-8 -*-
"""
auth.py — Google Earth Engine authentication.

Single authentication entry point shared by every GEE-backed download
(serves_gee, imerg_gee).  Credential resolution order:
  1. GOOGLE_APPLICATION_CREDENTIALS service-account file (or a key.json
     found next to this package, at the package root, or in the CWD)
  2. GEE_SERVICE_ACCOUNT_KEY inline JSON
  3. Earth Engine default credentials, then the interactive flow.
"""

import json
import logging
import os

try:
    import ee
    GEE_AVAILABLE = True
except ImportError:
    GEE_AVAILABLE = False

logger = logging.getLogger(__name__)


def authenticate(project=None):
    """Initialize GEE with the best available credentials."""
    proj = project or os.environ.get('GEE_PROJECT')
    init_kw = {'project': proj} if proj else {}

    sa_path = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')
    if not sa_path:
        # Non-interactive shells (conda run, HPC) don't source ~/.bashrc, so the
        # env var may be absent.  Fall back to a key.json next to this module,
        # at the package root, or in the current working directory.
        _here = os.path.dirname(os.path.abspath(__file__))
        for _candidate in (os.path.join(_here, 'key.json'),
                           os.path.join(os.path.dirname(_here), 'key.json'),
                           os.path.join(os.getcwd(), 'key.json')):
            if os.path.isfile(_candidate):
                sa_path = _candidate
                break
    if sa_path and os.path.isfile(sa_path):
        try:
            credentials = ee.ServiceAccountCredentials(None, sa_path)
            ee.Initialize(credentials, **init_kw)
            logger.info("GEE authenticated via GOOGLE_APPLICATION_CREDENTIALS")
            return True
        except Exception as exc:
            logger.warning("Service account auth failed: %s", exc)

    sa_json = os.environ.get('GEE_SERVICE_ACCOUNT_KEY')
    if sa_json:
        try:
            key_data = json.loads(sa_json)
            credentials = ee.ServiceAccountCredentials(
                key_data['client_email'], key_data=sa_json
            )
            ee.Initialize(credentials, **init_kw)
            logger.info("GEE authenticated via GEE_SERVICE_ACCOUNT_KEY")
            return True
        except Exception as exc:
            logger.warning("Inline service account auth failed: %s", exc)

    try:
        ee.Initialize(**init_kw)
        logger.info("GEE authenticated via default credentials")
        return True
    except Exception:
        try:
            ee.Authenticate()
            ee.Initialize(**init_kw)
            logger.info("GEE authenticated via interactive flow")
            return True
        except Exception as exc:
            logger.warning("GEE authentication failed: %s", exc)
            return False
