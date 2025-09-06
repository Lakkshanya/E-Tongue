import React, { useState } from "react";
import Login from "./pages/Login";
import Dashboard from "./components/Dashboard";

const App: React.FC = () => {
  const [loggedIn, setLoggedIn] = useState(false);

  if (!loggedIn) return <Login onLogin={() => setLoggedIn(true)} />;

  return <Dashboard />;
};

export default App;
