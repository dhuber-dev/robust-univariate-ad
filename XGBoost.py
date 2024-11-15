class XGBoostModel:
    def __init__(self, num_classes):
        """
        Initialize the XGBoostModel for multi-class classification.
        """
        self.num_classes = num_classes
        self.model = self._build_model()

    def _build_model(self):
        """
        Build and compile an XGBoost model for feature-based multi-class classification.
        """
        model = XGBClassifier(
            objective='multi:softmax',
            num_class=self.num_classes,
            max_depth=6,
            learning_rate=0.1,
            n_estimators=100,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1,
            reg_alpha=0.5,
            verbosity=1
        )
        return model

    def get_model(self):
        """
        Return the XGBoost model.
        """
        return self.model