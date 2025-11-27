"""
DEPRECATED: This module is deprecated and will be removed in a future version.

The modules in this package have been migrated to the new backend structure:

- modules/data_fetcher.py  →  backend/src/app/services/data_fetcher.py
- modules/database.py      →  backend/src/app/db/ (models + session)
- modules/model_trainer.py →  backend/src/app/services/model_trainer.py
- modules/email_notifier.py →  backend/src/app/services/email_service.py
- modules/orchestrator.py  →  backend/src/app/services/ (split into services)

For new development, please use the backend package instead:
    from app.services import fetch_stock_data, train_prediction_model

The legacy modules remain for backward compatibility with:
- train_local.py script
- Existing Airflow DAGs (until they're updated)
"""

import warnings

warnings.warn(
    "The 'modules' package is deprecated. Use 'backend.src.app.services' instead.",
    DeprecationWarning,
    stacklevel=2,
)

