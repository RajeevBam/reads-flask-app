"""Initialize Flask app."""
from flask import Flask
from flask_assets import Environment


def init_app():
    """Construct core Flask application with embedded Dash app."""
    app = Flask(__name__, instance_relative_config=False)
  

    with app.app_context():
        # Import parts of our core Flask app
        from . import routes

        # Import Dash application
        from .dashboard import init_dashboard
        

        app = init_dashboard(app)
        from .ExplainerDashboard.__init__ import init_explainer_dashboard
        app = init_explainer_dashboard(app)


        return app
