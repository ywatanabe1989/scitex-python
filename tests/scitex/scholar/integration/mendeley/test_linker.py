# Add your tests here

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/scholar/integration/mendeley/linker.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# Mendeley linker - Live synchronization with Mendeley.
# """
# 
# from typing import Optional
# import os
# 
# from scitex import logging
# from ..base import BaseLinker
# from .importer import MendeleyImporter
# from .exporter import MendeleyExporter
# from .mapper import MendeleyMapper
# 
# logger = logging.getLogger(__name__)
# 
# 
# class MendeleyLinker(BaseLinker):
#     """Live synchronization with Mendeley."""
# 
#     def __init__(
#         self,
#         app_id: Optional[str] = None,
#         app_secret: Optional[str] = None,
#         access_token: Optional[str] = None,
#         project: str = "default",
#         config=None,
#         sync_interval: int = 60,
#     ):
#         """Initialize Mendeley linker.
# 
#         Args:
#             app_id: Mendeley app ID
#             app_secret: Mendeley app secret
#             access_token: OAuth access token
#             project: Scholar project name
#             config: Optional ScholarConfig instance
#             sync_interval: Seconds between syncs
#         """
#         credentials = {
#             "app_id": app_id or os.getenv("MENDELEY_APP_ID"),
#             "app_secret": app_secret or os.getenv("MENDELEY_APP_SECRET"),
#             "access_token": access_token or os.getenv("MENDELEY_ACCESS_TOKEN"),
#         }
# 
#         super().__init__(
#             credentials=credentials,
#             project=project,
#             config=config,
#             sync_interval=sync_interval,
#         )
# 
#     def _create_importer(self) -> MendeleyImporter:
#         """Create importer instance."""
#         return MendeleyImporter(
#             app_id=self.credentials["app_id"],
#             app_secret=self.credentials["app_secret"],
#             access_token=self.credentials["access_token"],
#             project=self.project,
#             config=self.config,
#         )
# 
#     def _create_exporter(self) -> MendeleyExporter:
#         """Create exporter instance."""
#         return MendeleyExporter(
#             app_id=self.credentials["app_id"],
#             app_secret=self.credentials["app_secret"],
#             access_token=self.credentials["access_token"],
#             project=self.project,
#             config=self.config,
#         )
# 
#     def _create_mapper(self) -> MendeleyMapper:
#         """Create mapper instance."""
#         return MendeleyMapper(config=self.config)
# 
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/scholar/integration/mendeley/linker.py
# --------------------------------------------------------------------------------
