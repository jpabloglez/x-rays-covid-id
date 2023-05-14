import React, { Component } from 'react';

var stylingObject = {
    div: {
        color: "white",
        border: "0.1rem solid red",
        padding: "1rem 1rem",
        margin: "1rem 1rem",

      
    }, input: {
      margin: "2px",
      padding: "5px"
    }
  }

class Shopping extends React.Component {

    static num = 12345.6789;

    constructor() {
        super();
        this.state = {
          name: "React",
        };

    }

    render() {
      return (
        /*<div className="border-2" style={stylingObject.div}>*/
        <div className="rounded-md border-2 border-red-300 bg-red-100">
          <h1>Shopping List for {this.props.name}</h1>
          <ul>
            <li>Instagram</li>
            <li>WhatsApp</li>
            <li>Oculus</li>
          </ul>
        </div>
      );
    }
  }

export default Shopping;
