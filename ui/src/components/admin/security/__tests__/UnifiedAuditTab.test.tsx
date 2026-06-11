import React from 'react';
import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { UnifiedAuditTab } from '../UnifiedAuditTab';

describe('UnifiedAuditTab', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    (global.fetch as jest.Mock) = jest.fn().mockResolvedValue({
      ok: true,
      json: async () => ({
        records: [],
        total: 0,
        page: 1,
        limit: 30,
      }),
    });
  });

  it('shows admin required message when not admin (fetch still runs before gate)', async () => {
    render(<UnifiedAuditTab isAdmin={false} />);
    expect(
      screen.getByText(/Admin access required to view audit events/i)
    ).toBeInTheDocument();
    await waitFor(() => {
      expect(global.fetch).toHaveBeenCalled();
    });
  });

  it('loads audit events when admin', async () => {
    (global.fetch as jest.Mock).mockResolvedValue({
      ok: true,
      json: async () => ({
        records: [
          {
            id: '1',
            ts: new Date().toISOString(),
            type: 'auth',
            outcome: 'allow',
            action: 'admin_ui#view',
            tenant_id: 'default',
            subject_hash: 'h',
            correlation_id: 'c',
            source: 'bff',
          },
        ],
        total: 1,
        page: 1,
        limit: 30,
      }),
    });

    render(<UnifiedAuditTab isAdmin />);

    await waitFor(() => {
      expect(global.fetch).toHaveBeenCalled();
    });

    expect(await screen.findByText(/RBAC Audit Log/i)).toBeInTheDocument();
    expect(screen.getByRole('option', { name: /^All$/i })).toBeInTheDocument();
    expect(screen.queryByText(/Default view hides routine admin page-view checks/i)).not.toBeInTheDocument();
    expect(screen.queryByRole('option', { name: /^All types$/i })).not.toBeInTheDocument();
    expect(screen.getByText(/admin_ui#view/i)).toBeInTheDocument();
    expect(screen.getByText(/webui_backend/i)).toBeInTheDocument();
    expect(screen.queryByText(/^bff$/i)).not.toBeInTheDocument();
  });

  it('renders OpenFGA ReBAC audit events as their own type', async () => {
    (global.fetch as jest.Mock).mockResolvedValue({
      ok: true,
      json: async () => ({
        records: [
          {
            id: '1',
            ts: new Date().toISOString(),
            type: 'openfga_rebac',
            outcome: 'allow',
            action: 'agent#use',
            tenant_id: 'default',
            subject_hash: 'h',
            correlation_id: 'c',
            source: 'bff',
            pdp: 'openfga',
            resource_ref: 'user:alice can_use agent:default',
          },
        ],
        total: 1,
        page: 1,
        limit: 30,
      }),
    });

    render(<UnifiedAuditTab isAdmin />);

    expect(await screen.findByText(/agent#use/i)).toBeInTheDocument();
    expect(screen.getAllByText(/OpenFGA ReBAC/i).length).toBeGreaterThan(0);
  });

  it('blends bridge OpenFGA decisions into the audit table without rendering trace links', async () => {
    (global.fetch as jest.Mock).mockImplementation((url: string) => {
      return Promise.resolve({
        ok: true,
        json: async () => ({
          records: [
            {
              id: '1',
              ts: new Date().toISOString(),
              type: 'openfga_rebac',
              outcome: 'allow',
              action: 'mcp#can_call',
              tenant_id: 'default',
              subject_hash: 'h',
              correlation_id: 'c',
              source: 'openfga_authz_bridge',
              component: 'agent_gateway',
              pdp: 'openfga',
              resource_ref: 'user:alice can_call mcp_gateway:list',
              trace_id: '0123456789abcdef0123456789abcdef',
            },
          ],
          total: 1,
          page: 1,
          limit: 30,
        }),
      });
    });

    render(<UnifiedAuditTab isAdmin />);

    expect(await screen.findByText(/mcp#can_call/i)).toBeInTheDocument();
    expect(screen.getByText(/openfga_authz_bridge/i)).toBeInTheDocument();
    fireEvent.click(screen.getByText(/mcp#can_call/i));
    expect(screen.queryByText(/^Trace:/i)).not.toBeInTheDocument();
  });

  it('renders cas_grant policy-change events with caller and grantee context', async () => {
    (global.fetch as jest.Mock).mockResolvedValue({
      ok: true,
      json: async () => ({
        records: [
          {
            ts: new Date().toISOString(),
            type: 'cas_grant',
            outcome: 'success',
            action: 'use',
            tenant_id: 'acme',
            subject_hash: 'hash-caller',
            correlation_id: 'grant-corr',
            source: 'cas',
            caller_ref: 'user:alice',
            grantee_ref: 'team:eng',
            operation: 'grant',
            resource_ref: 'agent:platform-engineer',
            component: 'cas',
            pdp: 'openfga',
          },
          {
            ts: new Date().toISOString(),
            type: 'cas_grant',
            outcome: 'error',
            action: 'use',
            tenant_id: 'acme',
            subject_hash: 'hash-caller',
            correlation_id: 'grant-deny-corr',
            source: 'cas',
            caller_ref: 'user:bob',
            grantee_ref: 'team:eng',
            operation: 'revoke',
            reason_code: 'NO_CAPABILITY',
            resource_ref: 'agent:platform-engineer',
          },
        ],
        total: 2,
        page: 1,
        limit: 30,
      }),
    });

    render(<UnifiedAuditTab isAdmin />);

    expect(await screen.findByText(/user:alice/i)).toBeInTheDocument();
    expect(screen.getByText(/user:bob/i)).toBeInTheDocument();
    expect(screen.getAllByText(/Policy Grant\/Revoke/i).length).toBeGreaterThan(0);
    expect(screen.getByRole('option', { name: /Policy Grant\/Revoke/i })).toBeInTheDocument();

    fireEvent.click(screen.getByText(/user:alice/i));
    expect(await screen.findByText(/Grantee:/i)).toBeInTheDocument();
    expect(screen.getByText(/team:eng/i)).toBeInTheDocument();
    expect(screen.getByText(/Operation:/i)).toBeInTheDocument();
    expect(screen.getByText(/^grant$/i)).toBeInTheDocument();
  });
});
