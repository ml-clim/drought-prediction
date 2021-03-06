import sys

sys.path.append("../..")
from src.models import (
    Persistence,
    LinearRegression,
    LinearNetwork,
    RecurrentNetwork,
    EARecurrentNetwork,
    load_model,
    GBDT,
)
from src.analysis import all_explanations_for_file
from scripts.utils import get_data_path


def parsimonious(experiment="one_month_forecast",):
    predictor = Persistence(get_data_path(), experiment=experiment)
    predictor.evaluate(save_preds=True)


def regression(
    experiment="one_month_forecast",
    include_pred_month=True,
    surrounding_pixels=None,
    explain=False,
    static="features",
    ignore_vars=None,
    predict_delta=False,
    spatial_mask=None,
    include_latlons=False,
):
    predictor = LinearRegression(
        get_data_path(),
        experiment=experiment,
        include_pred_month=include_pred_month,
        surrounding_pixels=surrounding_pixels,
        static=static,
        ignore_vars=ignore_vars,
        predict_delta=predict_delta,
        spatial_mask=spatial_mask,
        include_latlons=include_latlons,
    )
    predictor.train()
    predictor.evaluate(save_preds=True)

    # mostly to test it works
    if explain:
        predictor.explain(save_shap_values=True)


def linear_nn(
    experiment="one_month_forecast",
    include_pred_month=True,
    surrounding_pixels=None,
    explain=False,
    static="features",
    ignore_vars=None,
    num_epochs=50,
    early_stopping=5,
    layer_sizes=[100],
    predict_delta=False,
    spatial_mask=None,
    include_latlons=False,
):
    predictor = LinearNetwork(
        layer_sizes=layer_sizes,
        data_folder=get_data_path(),
        experiment=experiment,
        include_pred_month=include_pred_month,
        surrounding_pixels=surrounding_pixels,
        static=static,
        ignore_vars=ignore_vars,
        predict_delta=predict_delta,
        spatial_mask=spatial_mask,
        include_latlons=include_latlons,
    )
    predictor.train(num_epochs=num_epochs, early_stopping=early_stopping)
    predictor.evaluate(save_preds=True)
    predictor.save_model()

    if explain:
        _ = predictor.explain(save_shap_values=True)


def rnn(
    experiment="one_month_forecast",
    include_pred_month=True,
    surrounding_pixels=None,
    explain=False,
    static="features",
    ignore_vars=None,
    num_epochs=50,
    early_stopping=5,
    hidden_size=128,
    predict_delta=False,
    spatial_mask=None,
    include_latlons=False,
    normalize_y=True,
    include_prev_y=True,
):
    predictor = RecurrentNetwork(
        hidden_size=hidden_size,
        data_folder=get_data_path(),
        experiment=experiment,
        include_pred_month=include_pred_month,
        surrounding_pixels=surrounding_pixels,
        static=static,
        ignore_vars=ignore_vars,
        predict_delta=predict_delta,
        spatial_mask=spatial_mask,
        include_latlons=include_latlons,
        normalize_y=normalize_y,
        include_prev_y=include_prev_y,
    )
    predictor.train(num_epochs=num_epochs, early_stopping=early_stopping)
    predictor.evaluate(save_preds=True)
    predictor.save_model()

    if explain:
        _ = predictor.explain(save_shap_values=True)


def earnn(
    experiment="one_month_forecast",
    include_pred_month=True,
    surrounding_pixels=None,
    pretrained=False,
    explain=False,
    static="features",
    ignore_vars=None,
    num_epochs=50,
    early_stopping=5,
    static_embedding_size=10,
    hidden_size=128,
    predict_delta=False,
    spatial_mask=None,
    include_latlons=False,
    normalize_y=True,
    include_prev_y=True,
):
    data_path = get_data_path()

    if not pretrained:
        predictor = EARecurrentNetwork(
            hidden_size=hidden_size,
            data_folder=data_path,
            experiment=experiment,
            include_pred_month=include_pred_month,
            surrounding_pixels=surrounding_pixels,
            static=static,
            static_embedding_size=static_embedding_size,
            ignore_vars=ignore_vars,
            predict_delta=predict_delta,
            spatial_mask=spatial_mask,
            include_latlons=include_latlons,
            normalize_y=normalize_y,
            include_prev_y=include_prev_y,
        )
        predictor.train(num_epochs=num_epochs, early_stopping=early_stopping)
        predictor.evaluate(save_preds=True)
        predictor.save_model()
    else:
        predictor = load_model(data_path / f"models/{experiment}/ealstm/model.pt")

    if explain:
        test_file = data_path / f"features/{experiment}/test/2018_3"
        assert test_file.exists()
        all_explanations_for_file(test_file, predictor, batch_size=100)


def gbdt(
    experiment="one_month_forecast",
    include_pred_month=True,
    surrounding_pixels=None,
    pretrained=True,
    explain=False,
    static="features",
    ignore_vars=None,
    # predict_delta=False,
    spatial_mask=None,
    include_latlons=False,
):
    data_path = get_data_path()

    # initialise, train and save GBDT model
    predictor = GBDT(
        data_folder=data_path,
        experiment=experiment,
        include_pred_month=include_pred_month,
        surrounding_pixels=surrounding_pixels,
        static=static,
        ignore_vars=ignore_vars,
        spatial_mask=spatial_mask,
        include_latlons=include_latlons,
    )
    predictor.train(early_stopping=5)
    predictor.evaluate(save_preds=True)
    predictor.save_model()
