BACCARAT H2O — RENDER READY PACKAGE

WHAT THIS PACKAGE DOES
- Hosts your latest Baccarat app at /
- Hosts the older H2O training interface at /training
- Runs the H2O/FastAPI server
- Uses /score for H2O predictions
- Uses /train for model training
- Uses /health to confirm the server is running

EASIEST DEPLOYMENT
1. Create a GitHub account if you do not already have one.
2. Create a new repository, for example: baccarat-h2o-ai
3. Upload ALL files from this package into the repository root.
4. Sign in to Render and connect your GitHub account.
5. Create a new Web Service from that repository.
6. Render should detect the Dockerfile. Choose Docker if asked.
7. Deploy.
8. When Render gives you an address such as:
   https://baccarat-h2o-ai.onrender.com
   open that address on your iPad or iPhone.
9. Your main Baccarat app will be at the main address.
10. The training page will be at:
    https://YOUR-RENDER-ADDRESS/training
11. Server test:
    https://YOUR-RENDER-ADDRESS/health

IMPORTANT
- A brand-new server has no trained model yet.
- Train the model before expecting /score predictions.
- The H2O model estimates statistical patterns; it cannot guarantee future baccarat outcomes.
