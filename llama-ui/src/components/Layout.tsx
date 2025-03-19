import { Box, Container, Tab, Tabs } from '@mui/material';
import { useState } from 'react';
import ModelManager from './ModelManager';
import Chat from './Chat';
import SystemMetrics from './SystemMetrics';

interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}

function TabPanel(props: TabPanelProps) {
  const { children, value, index, ...other } = props;

  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`simple-tabpanel-${index}`}
      aria-labelledby={`simple-tab-${index}`}
      {...other}
    >
      {value === index && (
        <Box sx={{ p: 3 }}>
          {children}
        </Box>
      )}
    </div>
  );
}

function Layout() {
  const [tabValue, setTabValue] = useState(0);

  const handleChange = (_event: React.SyntheticEvent, newValue: number) => {
    setTabValue(newValue);
  };

  return (
    <Container maxWidth="lg" sx={{ mt: 4 }}>
      <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
        <Tabs value={tabValue} onChange={handleChange} aria-label="basic tabs">
          <Tab label="Models" />
          <Tab label="Chat" />
          <Tab label="System" />
        </Tabs>
      </Box>
      <TabPanel value={tabValue} index={0}>
        <ModelManager />
      </TabPanel>
      <TabPanel value={tabValue} index={1}>
        <Chat />
      </TabPanel>
      <TabPanel value={tabValue} index={2}>
        <SystemMetrics />
      </TabPanel>
    </Container>
  );
}

export default Layout; 