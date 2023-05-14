import React from 'react';
import { Link } from 'react-router-dom';

const About = () => {
    return (
        <div>

            <div className="columns-2">
            <a href="https://vitejs.dev" target="_blank">
            <img src="/vite.svg" className="logo" alt="Vite logo" />
            </a>
            <a href="https://reactjs.org" target="_blank">
            <img src={reactLogo} className="logo react" alt="React logo" />
            </a>
            </div>
            <h1>Vite + React</h1>
            <div className="card">
            <button onClick={() => setCount((count) => count + 1)}>
                count is {count}
            </button>
            <p>
                Edit <code>src/App.tsx</code> and save to test HMR
            </p>
            </div>
            <div className="card">
                <p> <Link to="/about">About</Link> </p>
            </div>
        </div>
    );
}