pybabel extract -F babel.cfg -o translations/messages.pot .
#pybabel init -i messages.pot -d translations -l fr
pybabel update -i translations/messages.pot -d translations
pybabel compile -d translations