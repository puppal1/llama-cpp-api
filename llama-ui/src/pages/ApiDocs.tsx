import React from 'react';
import { RedocStandalone } from 'redoc';

const ApiDocs: React.FC = () => {
  return (
    <div style={{ height: '100vh' }}>
      <RedocStandalone
        specUrl="http://localhost:8000/api/v2/docs/openapi.yaml"
        options={{
          theme: {
            colors: {
              primary: { main: '#50fa7b' },
            },
            typography: {
              fontFamily: 'JetBrains Mono, monospace',
              headings: {
                fontFamily: 'JetBrains Mono, monospace',
              },
              code: {
                fontFamily: 'JetBrains Mono, monospace',
              },
            },
            sidebar: {
              backgroundColor: '#282a36',
              textColor: '#f8f8f2',
            },
            rightPanel: {
              backgroundColor: '#282a36',
              textColor: '#f8f8f2',
            },
          },
          hideDownloadButton: true,
          disableSearch: false,
          expandResponses: "200",
          jsonSampleExpandLevel: 3,
          requiredPropsFirst: true,
          showExtensions: false,
          sortPropsAlphabetically: false,
          hideHostname: false,
          menuToggle: true,
        }}
      />
    </div>
  );
};

export default ApiDocs;