import React from 'react';
import { RedocStandalone } from 'redoc';

const ApiDocs = () => {
  return (
    <RedocStandalone
      specUrl="/api/v2/docs/openapi.yaml"
      options={{
        theme: {
          colors: {
            primary: {
              main: '#4caf50'
            },
            text: {
              primary: '#ffffff',
              secondary: '#9e9e9e'
            },
            http: {
              get: '#4caf50',
              post: '#2196f3',
              delete: '#f44336',
              put: '#ff9800',
              basic: '#9e9e9e',
              link: '#2196f3',
              head: '#9e9e9e'
            },
            responses: {
              success: { color: '#4caf50' },
              error: { color: '#f44336' },
              redirect: { color: '#ff9800' },
              info: { color: '#2196f3' }
            },
            border: {
              dark: '#333333',
              light: '#666666'
            }
          },
          typography: {
            fontSize: '16px',
            headings: {
              fontWeight: 'bold'
            }
          },
          sidebar: {
            backgroundColor: '#1a1a1a',
            textColor: '#ffffff',
            activeTextColor: '#4caf50'
          },
          rightPanel: {
            backgroundColor: '#262626'
          },
          codeBlock: {
            backgroundColor: '#1e1e1e'
          },
          schema: {
            nestedBackground: '#262626',
            typeNameColor: '#4caf50',
            typeTitleColor: '#ffffff'
          }
        },
        expandResponses: 'all',
        expandSingleSchemaField: true,
        hideDownloadButton: true,
        hideFab: true,
        hideHostname: true,
        hideLoading: true,
        hideSchemaPattern: false,
        menuToggle: true,
        nativeScrollbars: true,
        pathInMiddlePanel: true,
        requiredPropsFirst: true,
        showExtensions: false,
        sortEnumValuesAlphabetically: true,
        sortOperationsAlphabetically: false,
        sortPropsAlphabetically: true
      }}
    />
  );
};

export default ApiDocs;